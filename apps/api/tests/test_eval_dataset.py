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


if __name__ == "__main__":
    unittest.main()
