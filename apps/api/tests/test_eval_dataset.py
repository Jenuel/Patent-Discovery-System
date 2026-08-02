"""Unit tests for evaluation.dataset (run from apps/api: python -m unittest discover tests)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluation.dataset import QueryCase, load_query_cases, parse_query_cases

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "fixtures"


class ParseQueryCasesTests(unittest.TestCase):
    def test_normal_case(self):
        lines = [
            json.dumps({
                "query_id": "q1",
                "query": "cathode material",
                "metadata_filter": {"filing_year": {"$gte": 2018}},
                "relevant_patent_ids": ["US1", "US2"],
                "relevant_chunk_ids": {"US1::abstract::0000": 2, "US2::claim::0001": 1},
                "notes": "example",
            })
        ]
        cases = parse_query_cases(lines)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.query_id, "q1")
        self.assertEqual(case.query, "cathode material")
        self.assertEqual(case.metadata_filter, {"filing_year": {"$gte": 2018}})
        self.assertEqual(case.relevant_patent_ids, frozenset({"US1", "US2"}))
        self.assertEqual(case.relevant_chunk_ids, {"US1::abstract::0000": 2, "US2::claim::0001": 1})
        self.assertEqual(case.relevant_chunk_id_set, frozenset({"US1::abstract::0000", "US2::claim::0001"}))
        self.assertEqual(case.notes, "example")

    def test_defaulted_optional_fields(self):
        lines = [json.dumps({"query_id": "q1", "query": "minimal query"})]
        cases = parse_query_cases(lines)
        expected = QueryCase(query_id="q1", query="minimal query")
        self.assertEqual(cases[0], expected)

    def test_blank_lines_are_skipped(self):
        lines = ["", "   ", json.dumps({"query_id": "q1", "query": "x"}), ""]
        cases = parse_query_cases(lines)
        self.assertEqual(len(cases), 1)

    def test_malformed_json_raises_value_error_with_line_number(self):
        lines = [
            json.dumps({"query_id": "q1", "query": "ok"}),
            "{not valid json",
        ]
        with self.assertRaises(ValueError) as ctx:
            parse_query_cases(lines)
        self.assertIn("line 2", str(ctx.exception))

    def test_missing_required_field_raises_value_error(self):
        lines = [json.dumps({"query": "no id here"})]
        with self.assertRaises(ValueError) as ctx:
            parse_query_cases(lines)
        self.assertIn("query_id", str(ctx.exception))


class FixtureConsistencyTests(unittest.TestCase):
    """Guards evaluation/fixtures/queries.jsonl and corpus_chunks.jsonl against drift."""

    def test_relevant_chunk_ids_reference_real_chunks(self):
        cases = load_query_cases(FIXTURES_DIR / "queries.jsonl")

        known_chunk_ids = set()
        with open(FIXTURES_DIR / "corpus_chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                known_chunk_ids.add(json.loads(line)["id"])

        for case in cases:
            for chunk_id in case.relevant_chunk_ids:
                self.assertIn(
                    chunk_id, known_chunk_ids,
                    f"query {case.query_id!r} references unknown chunk id {chunk_id!r}",
                )

    def test_relevant_patent_ids_reference_real_patents(self):
        cases = load_query_cases(FIXTURES_DIR / "queries.jsonl")

        known_patent_ids = set()
        with open(FIXTURES_DIR / "corpus_patents.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                known_patent_ids.add(json.loads(line)["id"])

        for case in cases:
            for patent_id in case.relevant_patent_ids:
                self.assertIn(
                    patent_id, known_patent_ids,
                    f"query {case.query_id!r} references unknown patent id {patent_id!r}",
                )


class PooledFixtureTests(unittest.TestCase):
    """Guards queries_pooled.jsonl — the TREC-pooled re-judgement of the labels.

    Its patent ids come from the live 6,000-patent corpus, not the 131-patent
    sample in corpus_patents.jsonl, so it cannot use FixtureConsistencyTests'
    membership checks. What is guarded instead is the three defects the
    re-judgement exists to fix.
    """

    @classmethod
    def setUpClass(cls):
        cls.path = FIXTURES_DIR / "queries_pooled.jsonl"
        cls.raw = [
            json.loads(line)
            for line in cls.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.cases = load_query_cases(cls.path)

    def test_loads_with_the_same_parser_as_the_original(self):
        self.assertEqual(len(self.cases), len(self.raw))
        self.assertEqual(
            {c.query_id for c in self.cases}, {r["query_id"] for r in self.raw}
        )

    def test_no_abstract_chunks_in_claim_ground_truth(self):
        """claims_hybrid stores only claim chunks.

        The original labels marked ``<patent>::abstract::0000`` relevant for
        every relevant patent — half of all chunk labels — but Stage 2 searches
        a collection that contains no abstracts, so those could never be
        retrieved. Scoring against them capped claim recall at 0.5.
        """
        for case in self.cases:
            for chunk_id in case.relevant_chunk_ids:
                self.assertNotIn(
                    "::abstract::", chunk_id,
                    f"{case.query_id}: {chunk_id} is unretrievable from claims_hybrid",
                )

    def test_only_grade_2_labels_survive(self):
        """Grade 1 failed audit at a 52.5% overturn rate and was dropped.

        Re-admitting it would put labels back that are wrong about half the
        time — worse than the false negatives the pooling pass removed.
        """
        for rec in self.raw:
            for pid, grade in rec.get("patent_grades", {}).items():
                self.assertEqual(grade, 2, f"{rec['query_id']}: {pid} graded {grade}")
            for cid, grade in rec["relevant_chunk_ids"].items():
                self.assertEqual(grade, 2, f"{rec['query_id']}: {cid} graded {grade}")

    def test_every_query_keeps_ground_truth(self):
        """Dropping grade 1 must not leave a query unscoreable.

        ``aggregate`` excludes empty-ground-truth queries from its means, so a
        query emptied by the filter would vanish from the report rather than
        score zero.
        """
        for case in self.cases:
            self.assertTrue(case.relevant_patent_ids, f"{case.query_id} has no patents")
            self.assertTrue(case.relevant_chunk_ids, f"{case.query_id} has no chunks")

    def test_patent_grades_agree_with_relevant_patent_ids(self):
        for rec in self.raw:
            self.assertEqual(
                sorted(rec["patent_grades"]), sorted(rec["relevant_patent_ids"]),
                f"{rec['query_id']}: grade map and id list disagree",
            )

    def test_every_label_records_how_it_was_derived(self):
        """Provenance is the point: these labels are model-judged, not human."""
        allowed = {
            "prior_cpc_family", "prior_abstract_verified", "prior_rejected",
            "llm_pooled", "dropped_unretrievable",
        }
        for rec in self.raw:
            prov = rec["provenance"]
            self.assertIn("judged_by", prov)
            self.assertIn("audit", prov)
            for pid in rec["relevant_patent_ids"]:
                self.assertIn(prov["patents"].get(pid), allowed, f"{rec['query_id']}/{pid}")
            for cid in rec["relevant_chunk_ids"]:
                self.assertIn(prov["chunks"].get(cid), allowed, f"{rec['query_id']}/{cid}")

    def test_chunks_belong_to_a_relevant_patent(self):
        """A relevant claim whose patent is not relevant would be incoherent."""
        for rec in self.raw:
            patents = set(rec["relevant_patent_ids"])
            for cid in rec["relevant_chunk_ids"]:
                self.assertIn(
                    cid.split("::")[0], patents,
                    f"{rec['query_id']}: {cid} has no relevant parent patent",
                )

    def test_queries_match_the_original_fixture(self):
        """Same 20 queries and filters — only the labels were re-judged."""
        original = {c.query_id: c for c in load_query_cases(FIXTURES_DIR / "queries.jsonl")}
        self.assertEqual(set(original), {c.query_id for c in self.cases})
        for case in self.cases:
            self.assertEqual(case.query, original[case.query_id].query)
            self.assertEqual(case.metadata_filter, original[case.query_id].metadata_filter)


if __name__ == "__main__":
    unittest.main()
