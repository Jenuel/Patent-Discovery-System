"""Unit tests for evaluation.retrieval_eval's pure helpers
(run from apps/api: python -m unittest discover tests).

Only dedupe_patent_ids is exercised here — it is the sole piece of
evaluation.retrieval_eval with no Qdrant/embedder dependency, matching
FakeDense's style in test_hierarchical.py of isolating pure logic from I/O.
"""
from __future__ import annotations

import unittest

from app.services.retrieval.fusion import ScoredMatch
from evaluation.retrieval_eval import dedupe_patent_ids


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


if __name__ == "__main__":
    unittest.main()
