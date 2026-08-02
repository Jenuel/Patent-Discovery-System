"""Unit tests for retrieval fusion helpers (run from apps/api: python -m unittest discover tests)."""
from __future__ import annotations

import unittest

from app.services.retrieval.fusion import (
    ScoredMatch,
    to_result_dicts,
    to_scored_matches,
    weighted_rrf,
)


class ToScoredMatchesTests(unittest.TestCase):
    def test_converts_result_dicts(self):
        raw = [{"id": "US1", "score": 0.5, "metadata": {"patent_id": "US1"}}]
        matches = to_scored_matches(raw)
        self.assertEqual(matches, [ScoredMatch(id="US1", score=0.5, metadata={"patent_id": "US1"})])

    def test_missing_fields_get_defaults(self):
        matches = to_scored_matches([{}])
        self.assertEqual(matches, [ScoredMatch(id="", score=0.0, metadata={})])


class ToResultDictsTests(unittest.TestCase):
    """The weighted arm fuses in ScoredMatch space but must hand back dicts."""

    def test_round_trips_through_to_scored_matches(self):
        raw = [{"id": "US1", "score": 0.5, "metadata": {"patent_id": "US1"}}]
        self.assertEqual(to_result_dicts(to_scored_matches(raw)), raw)

    def test_empty(self):
        self.assertEqual(to_result_dicts([]), [])


class WeightedRrfTests(unittest.TestCase):
    """EVAL_ABLATION.md Tests 4 and 6 — client-side weighted fusion (RET-09)."""

    @staticmethod
    def _arm(*ids):
        return [ScoredMatch(id=i, score=1.0, metadata={"patent_id": i}) for i in ids]

    def test_single_arm_preserves_its_ordering(self):
        fused = weighted_rrf([(1.0, self._arm("a", "b", "c"))], k=60)
        self.assertEqual([m.id for m in fused], ["a", "b", "c"])

    def test_zero_weight_arm_is_excluded_entirely(self):
        """The 1:0 row of Test 6 must equal dense-only, not merely approach it."""
        fused = weighted_rrf(
            [(1.0, self._arm("a", "b")), (0.0, self._arm("z", "y"))], k=60
        )
        self.assertEqual([m.id for m in fused], ["a", "b"])

    def test_agreement_between_arms_beats_a_single_arms_top_hit(self):
        dense = self._arm("a", "shared")
        sparse = self._arm("b", "shared")
        fused = weighted_rrf([(1.0, dense), (1.0, sparse)], k=1)
        # shared: 1/(1+2) + 1/(1+2) = 0.667; a and b: 1/(1+1) = 0.5
        self.assertEqual(fused[0].id, "shared")

    def test_weight_shifts_the_balance_toward_the_heavier_arm(self):
        dense = self._arm("d1", "d2")
        sparse = self._arm("s1", "s2")
        even = weighted_rrf([(1.0, dense), (1.0, sparse)], k=60)
        skewed = weighted_rrf([(10.0, dense), (1.0, sparse)], k=60)
        self.assertEqual(even[0].id, "d1")  # tie broken by first-seen order
        self.assertEqual([m.id for m in skewed[:2]], ["d1", "d2"])

    def test_the_shipped_ratio_keeps_dense_ahead_of_a_sparse_only_hit(self):
        """0.9/0.1 is the RET-09 default; a sparse-only rank-1 must not lead."""
        fused = weighted_rrf(
            [(0.9, self._arm("d1", "d2")), (0.1, self._arm("s1", "s2"))], k=60
        )
        self.assertEqual([m.id for m in fused[:2]], ["d1", "d2"])

    def test_scores_are_the_fused_values_not_the_input_scores(self):
        fused = weighted_rrf([(1.0, self._arm("a"))], k=60)
        self.assertAlmostEqual(fused[0].score, 1.0 / 61)

    def test_metadata_survives_fusion(self):
        fused = weighted_rrf([(1.0, self._arm("a"))], k=60)
        self.assertEqual(fused[0].metadata, {"patent_id": "a"})

    def test_limit_truncates(self):
        fused = weighted_rrf([(1.0, self._arm("a", "b", "c"))], k=60, limit=2)
        self.assertEqual([m.id for m in fused], ["a", "b"])

    def test_ties_resolve_deterministically_to_first_seen_order(self):
        for _ in range(5):
            fused = weighted_rrf(
                [(1.0, self._arm("a", "b")), (1.0, self._arm("a", "b"))], k=60
            )
            self.assertEqual([m.id for m in fused], ["a", "b"])

    def test_an_empty_arm_leaves_the_other_arms_ordering_intact(self):
        """A term-less query yields no BM25 hits (RET-06); dense must survive."""
        fused = weighted_rrf([(0.9, self._arm("a", "b")), (0.1, [])], k=60)
        self.assertEqual([m.id for m in fused], ["a", "b"])

    def test_no_arms(self):
        self.assertEqual(weighted_rrf([], k=60), [])


if __name__ == "__main__":
    unittest.main()
