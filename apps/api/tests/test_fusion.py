"""Unit tests for RRF fusion (run from apps/api: python -m unittest discover tests)."""
from __future__ import annotations

import unittest

from app.services.retrieval.fusion import ScoredMatch, fuse_rrf, to_scored_matches


def _match(patent_id: str, score: float = 1.0) -> ScoredMatch:
    return ScoredMatch(id=patent_id, score=score, metadata={"patent_id": patent_id})


class FuseRrfTests(unittest.TestCase):
    def test_empty_sparse_degrades_to_dense(self):
        dense = [_match(f"US{i}") for i in range(5)]
        fused = fuse_rrf(dense, [])
        self.assertEqual([m.id for m in fused], [f"US{i}" for i in range(5)])

    def test_empty_dense_degrades_to_sparse(self):
        sparse = [_match(f"US{i}") for i in range(3)]
        fused = fuse_rrf([], sparse)
        self.assertEqual([m.id for m in fused], ["US0", "US1", "US2"])

    def test_both_empty_returns_empty(self):
        self.assertEqual(fuse_rrf([], []), [])

    def test_union_keeps_ids_unique_to_either_source(self):
        dense = [_match("US1"), _match("US2")]
        sparse = [_match("US3")]
        fused = fuse_rrf(dense, sparse)
        self.assertEqual({m.id for m in fused}, {"US1", "US2", "US3"})

    def test_overlapping_id_ranks_above_single_source_ids(self):
        dense = [_match("US1"), _match("US2")]
        sparse = [_match("US3"), _match("US2")]
        fused = fuse_rrf(dense, sparse, k=60)
        self.assertEqual(fused[0].id, "US2")
        # Fused score is the sum of both sources' RRF contributions
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 62)

    def test_top_k_limits_results(self):
        dense = [_match(f"US{i}") for i in range(10)]
        fused = fuse_rrf(dense, [], top_k=3)
        self.assertEqual(len(fused), 3)

    def test_falls_back_to_match_id_without_patent_id_metadata(self):
        dense = [ScoredMatch(id="chunk-7", score=1.0, metadata={})]
        fused = fuse_rrf(dense, [])
        self.assertEqual(fused[0].id, "chunk-7")

    def test_dense_metadata_preferred_on_overlap(self):
        dense = [ScoredMatch(id="d1", score=1.0, metadata={"patent_id": "US1", "src": "dense"})]
        sparse = [ScoredMatch(id="s1", score=1.0, metadata={"patent_id": "US1", "src": "sparse"})]
        fused = fuse_rrf(dense, sparse)
        self.assertEqual(fused[0].metadata["src"], "dense")


class ToScoredMatchesTests(unittest.TestCase):
    def test_converts_result_dicts(self):
        raw = [{"id": "US1", "score": 0.5, "metadata": {"patent_id": "US1"}}]
        matches = to_scored_matches(raw)
        self.assertEqual(matches, [ScoredMatch(id="US1", score=0.5, metadata={"patent_id": "US1"})])

    def test_missing_fields_get_defaults(self):
        matches = to_scored_matches([{}])
        self.assertEqual(matches, [ScoredMatch(id="", score=0.0, metadata={})])


if __name__ == "__main__":
    unittest.main()
