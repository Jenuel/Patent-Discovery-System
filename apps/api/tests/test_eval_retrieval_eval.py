"""Unit tests for evaluation.retrieval_eval's pure helpers
(run from apps/api: python -m unittest discover tests).

dedupe_patent_ids is the only piece of evaluation.retrieval_eval with no
Qdrant/embedder dependency, matching FakeDense's style in test_hierarchical.py
of isolating pure logic from I/O.

BuildPipelineSignatureTests additionally guards build_pipeline's *call
signatures* without constructing anything — see its docstring.
"""
from __future__ import annotations

import csv
import inspect
import io
import unittest

from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.fusion import ScoredMatch
from app.services.retrieval.hierarchical import HierarchicalConfig, HierarchicalRetriever
from evaluation.retrieval_eval import dedupe_patent_ids, union_fieldnames


def _match(id_: str, patent_id: str = None, score: float = 1.0) -> ScoredMatch:
    metadata = {"patent_id": patent_id} if patent_id is not None else {}
    return ScoredMatch(id=id_, score=score, metadata=metadata)


class DedupePatentIdsTests(unittest.TestCase):
    def test_order_preserving_dedup(self):
        matches = [
            _match("c1", "US1"),
            _match("c2", "US2"),
            _match("c3", "US1"),
            _match("c4", "US3"),
        ]
        self.assertEqual(dedupe_patent_ids(matches, limit=10), ["US1", "US2", "US3"])

    def test_limit_truncates(self):
        matches = [_match("c1", "US1"), _match("c2", "US2"), _match("c3", "US3")]
        self.assertEqual(dedupe_patent_ids(matches, limit=2), ["US1", "US2"])

    def test_matches_with_no_patent_id_are_skipped(self):
        matches = [_match("c1", None), _match("c2", "US1"), _match("c3", None)]
        self.assertEqual(dedupe_patent_ids(matches, limit=10), ["US1"])

    def test_empty_matches(self):
        self.assertEqual(dedupe_patent_ids([], limit=10), [])


class BuildPipelineSignatureTests(unittest.TestCase):
    """
    build_pipeline() cannot be called under unittest — it hits OpenAI, Qdrant
    and the environment. That gap let EVAL-03 ship: it passed `sparse=None` to
    a HierarchicalRetriever whose constructor had dropped that parameter in the
    Qdrant refactor, so every call raised TypeError and the harness could never
    run.

    inspect.signature().bind() closes the gap. It validates the argument names
    build_pipeline uses against the live constructors without instantiating
    anything, so a future signature change fails here instead of at runtime.
    """

    def test_hierarchical_retriever_accepts_build_pipeline_kwargs(self):
        inspect.signature(HierarchicalRetriever).bind(
            dense=object(), cfg=HierarchicalConfig()
        )

    def test_hierarchical_retriever_rejects_the_removed_sparse_kwarg(self):
        """Pins the exact regression: `sparse` is gone and must stay gone."""
        with self.assertRaises(TypeError):
            inspect.signature(HierarchicalRetriever).bind(
                dense=object(), sparse=None, cfg=HierarchicalConfig()
            )

    def test_dense_retriever_accepts_a_single_store_argument(self):
        inspect.signature(DenseRetriever).bind(object())


class UnionFieldnamesTests(unittest.TestCase):
    """
    EVAL-04: write_reports took CSV fieldnames from rows[0], but patent rows
    (k=3,5,10) and claim rows (k=5,10,20) carry different @k columns by design.
    csv.DictWriter raises on any key it was not told about, so writing the
    report failed with the DEFAULT k values every time.
    """

    def test_union_covers_columns_absent_from_the_first_row(self):
        rows = [
            {"level": "patent", "recall@3": 1.0, "recall@10": 0.5},
            {"level": "claim", "recall@10": 0.4, "recall@20": 0.9},
        ]
        names = union_fieldnames(rows)
        self.assertIn("recall@20", names, "column unique to a later row was dropped")
        self.assertIn("recall@3", names)

    def test_first_seen_order_is_preserved(self):
        rows = [{"query_id": "q1", "level": "patent", "mrr": 1.0},
                {"query_id": "q2", "level": "claim", "ndcg@20": 0.3}]
        self.assertEqual(
            union_fieldnames(rows), ["query_id", "level", "mrr", "ndcg@20"]
        )

    def test_no_duplicate_columns(self):
        rows = [{"a": 1, "b": 2}, {"b": 3, "a": 4}, {"a": 5}]
        self.assertEqual(union_fieldnames(rows), ["a", "b"])

    def test_empty_rows(self):
        self.assertEqual(union_fieldnames([]), [])

    def test_asymmetric_rows_actually_survive_a_dictwriter_round_trip(self):
        """The end-to-end guard: reproduces the exact original failure."""
        rows = [
            {"query_id": "q1", "level": "patent", "recall@3": 1.0, "recall@10": 0.5},
            {"query_id": "q1", "level": "claim", "recall@10": 0.4, "recall@20": 0.9},
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=union_fieldnames(rows), restval="")
        writer.writeheader()
        writer.writerows(rows)  # raised ValueError before the fix

        parsed = list(csv.DictReader(io.StringIO(buf.getvalue())))
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["recall@20"], "")      # patent row: blank, not missing
        self.assertEqual(parsed[1]["recall@20"], "0.9")


if __name__ == "__main__":
    unittest.main()
