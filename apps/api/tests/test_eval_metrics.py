"""Unit tests for evaluation.metrics (run from apps/api: python -m unittest discover tests)."""
from __future__ import annotations

import unittest

from evaluation.metrics import (
    aggregate,
    bootstrap_cis,
    dcg_at_k,
    evaluate_query,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_ceiling_at_k,
    reciprocal_rank,
)


class RecallPrecisionTests(unittest.TestCase):
    def test_recall_partial_hit(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"b", "z"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=4), 0.5)

    def test_recall_full_hit(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 1.0)

    def test_recall_k_smaller_than_relevant_set(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=1), 1 / 3)

    def test_recall_empty_relevant_set_is_zero(self):
        self.assertEqual(recall_at_k(["a"], set(), k=5), 0.0)

    def test_precision_partial_hit(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=4), 0.5)

    def test_precision_k_greater_than_retrieved_divides_by_actual_length(self):
        # Only 2 results retrieved even though k=10 — must not be penalized
        # as if the missing 8 slots were irrelevant results.
        retrieved = ["a", "b"]
        relevant = {"a"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=10), 0.5)

    def test_precision_no_results_is_zero(self):
        self.assertEqual(precision_at_k([], {"a"}, k=5), 0.0)


class HitRateTests(unittest.TestCase):
    def test_hit_rate_hit(self):
        self.assertEqual(hit_rate_at_k(["a", "b"], {"b"}, k=2), 1.0)

    def test_hit_rate_miss(self):
        self.assertEqual(hit_rate_at_k(["a", "b"], {"z"}, k=2), 0.0)

    def test_hit_rate_hit_outside_k_counts_as_miss(self):
        self.assertEqual(hit_rate_at_k(["a", "b", "c"], {"c"}, k=2), 0.0)


class ReciprocalRankTests(unittest.TestCase):
    def test_first_position_hit(self):
        self.assertAlmostEqual(reciprocal_rank(["a", "b"], {"a"}), 1.0)

    def test_later_position_hit(self):
        self.assertAlmostEqual(reciprocal_rank(["a", "b", "c"], {"c"}), 1 / 3)

    def test_no_hit(self):
        self.assertEqual(reciprocal_rank(["a", "b"], {"z"}), 0.0)


class NdcgTests(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        retrieved = ["a", "b", "c"]
        grades = {"a": 2, "b": 1, "c": 1}
        self.assertAlmostEqual(ndcg_at_k(retrieved, grades, k=3), 1.0)

    def test_reversed_ranking_is_lower_but_positive(self):
        # Hand-computed: grades{a:2,b:1,c:1}, retrieved reversed relative to ideal.
        retrieved = ["c", "b", "a"]
        grades = {"a": 2, "b": 1, "c": 1}
        import math

        dcg = 1 / math.log2(2) + 1 / math.log2(3) + 2 / math.log2(4)
        ideal_dcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        expected = dcg / ideal_dcg
        actual = ndcg_at_k(retrieved, grades, k=3)
        self.assertLess(actual, 1.0)
        self.assertGreater(actual, 0.0)
        self.assertAlmostEqual(actual, expected)

    def test_no_relevant_grades_is_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], {}, k=2), 0.0)

    def test_dcg_ignores_items_past_k(self):
        retrieved = ["z", "z", "a"]
        grades = {"a": 2}
        self.assertEqual(dcg_at_k(retrieved, grades, k=2), 0.0)


class EvaluateQueryTests(unittest.TestCase):
    def test_empty_relevant_set_marks_no_ground_truth(self):
        result = evaluate_query(["a", "b"], set(), k_values=(5,), query_id="q1")
        self.assertFalse(result.has_ground_truth)
        self.assertEqual(result.reciprocal_rank, 0.0)
        self.assertEqual(result.metrics_by_k[5].recall, 0.0)

    def test_non_empty_relevant_set_scores_normally(self):
        result = evaluate_query(["a", "b"], {"a"}, k_values=(5,), query_id="q1")
        self.assertTrue(result.has_ground_truth)
        self.assertAlmostEqual(result.reciprocal_rank, 1.0)
        self.assertAlmostEqual(result.metrics_by_k[5].recall, 1.0)


class AggregateTests(unittest.TestCase):
    def test_excludes_no_ground_truth_queries_from_mean(self):
        with_gt = evaluate_query(["a"], {"a"}, k_values=(5,), query_id="q1")
        without_gt = evaluate_query(["a"], set(), k_values=(5,), query_id="q2")

        summary = aggregate([with_gt, without_gt], k_values=(5,))

        self.assertEqual(summary["num_queries"], 2)
        self.assertEqual(summary["num_queries_with_ground_truth"], 1)
        self.assertAlmostEqual(summary["mrr"], 1.0)
        self.assertAlmostEqual(summary["recall@5"], 1.0)

    def test_all_queries_without_ground_truth_yields_zero_summary(self):
        without_gt = evaluate_query(["a"], set(), k_values=(5,), query_id="q1")
        summary = aggregate([without_gt], k_values=(5,))
        self.assertEqual(summary["num_queries_with_ground_truth"], 0)
        self.assertEqual(summary["mrr"], 0.0)
        self.assertEqual(summary["recall@5"], 0.0)


class RecallCeilingTests(unittest.TestCase):
    """recall@k cannot reach 1.0 when |R| > k; the ceiling says by how much."""

    def test_ceiling_is_k_over_relevant_when_relevant_set_is_larger(self):
        # The EVAL_BASELINE.md case: 8 relevant patents, k=3 -> 0.375 maximum.
        self.assertAlmostEqual(recall_ceiling_at_k(8, 3), 0.375)

    def test_ceiling_is_one_when_k_covers_the_relevant_set(self):
        self.assertAlmostEqual(recall_ceiling_at_k(8, 10), 1.0)
        self.assertAlmostEqual(recall_ceiling_at_k(8, 8), 1.0)

    def test_empty_relevant_set_is_zero_not_a_division_error(self):
        self.assertEqual(recall_ceiling_at_k(0, 5), 0.0)

    def test_a_perfect_retriever_scores_exactly_the_ceiling(self):
        """The ceiling is only meaningful if it is actually attainable."""
        relevant = {"a", "b", "c", "d", "e", "f", "g", "h"}
        perfect = evaluate_query(sorted(relevant), relevant, k_values=(3,))
        self.assertAlmostEqual(
            perfect.metrics_by_k[3].recall, recall_ceiling_at_k(len(relevant), 3)
        )


class BootstrapCiTests(unittest.TestCase):
    def test_interval_brackets_the_mean(self):
        values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        lo, hi = bootstrap_cis({"m": values}, num_samples=2000)["m"]
        self.assertLess(lo, sum(values) / len(values))
        self.assertGreater(hi, sum(values) / len(values))

    def test_zero_variance_input_gives_a_zero_width_interval(self):
        lo, hi = bootstrap_cis({"m": [0.5] * 10}, num_samples=500)["m"]
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_same_seed_reproduces_the_interval_exactly(self):
        values = [0.1, 0.9, 0.3, 0.7]
        first = bootstrap_cis({"m": values}, num_samples=500, seed=7)["m"]
        second = bootstrap_cis({"m": values}, num_samples=500, seed=7)["m"]
        self.assertEqual(first, second)

    def test_columns_share_one_set_of_resample_draws(self):
        """Identical columns must get identical intervals.

        If each column were resampled independently, two columns holding the
        same numbers would land on different intervals and could not be read
        against each other.
        """
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        cis = bootstrap_cis({"a": values, "b": list(values)}, num_samples=1000)
        self.assertEqual(cis["a"], cis["b"])

    def test_wider_confidence_gives_a_wider_interval(self):
        values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        narrow = bootstrap_cis({"m": values}, num_samples=2000, confidence=0.50)["m"]
        wide = bootstrap_cis({"m": values}, num_samples=2000, confidence=0.99)["m"]
        self.assertLess(wide[0], narrow[0])
        self.assertGreater(wide[1], narrow[1])

    def test_single_observation_collapses_onto_itself(self):
        lo, hi = bootstrap_cis({"m": [0.42]}, num_samples=100)["m"]
        self.assertAlmostEqual(lo, 0.42)
        self.assertAlmostEqual(hi, 0.42)

    def test_no_columns_is_empty_not_an_error(self):
        self.assertEqual(bootstrap_cis({}), {})

    def test_ragged_columns_are_rejected(self):
        """A short column would silently misalign query i across metrics."""
        with self.assertRaises(ValueError):
            bootstrap_cis({"a": [0.1, 0.2], "b": [0.3]}, num_samples=10)

    def test_invalid_confidence_is_rejected(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with self.assertRaises(ValueError, msg=str(bad)):
                bootstrap_cis({"m": [0.1, 0.2]}, num_samples=10, confidence=bad)


class AggregateIntervalTests(unittest.TestCase):
    """Item 4: the headline numbers must carry their own error bars."""

    def _summary(self, **kwargs):
        results = [
            evaluate_query(["a", "x"], {"a"}, k_values=(3,), query_id="q1"),
            evaluate_query(["x", "b"], {"b"}, k_values=(3,), query_id="q2"),
            evaluate_query(["x", "y"], {"c"}, k_values=(3,), query_id="q3"),
        ]
        return aggregate(results, k_values=(3,), **kwargs)

    def test_every_reported_mean_has_an_interval(self):
        summary = self._summary(bootstrap_samples=500)
        for metric in ("mrr", "recall@3", "precision@3", "hit_rate@3", "ndcg@3"):
            self.assertIn(f"{metric}_ci_lo", summary, metric)
            self.assertIn(f"{metric}_ci_hi", summary, metric)
            self.assertLessEqual(summary[f"{metric}_ci_lo"], summary[metric])
            self.assertGreaterEqual(summary[f"{metric}_ci_hi"], summary[metric])

    def test_recall_ceiling_is_reported_beside_recall(self):
        summary = self._summary(bootstrap_samples=0)
        self.assertIn("recall_ceiling@3", summary)
        # |R| = 1 for all three queries, so k=3 covers them entirely.
        self.assertAlmostEqual(summary["recall_ceiling@3"], 1.0)

    def test_ceiling_falls_below_one_when_relevant_sets_outgrow_k(self):
        relevant = {"a", "b", "c", "d"}
        results = [evaluate_query(["a"], relevant, k_values=(2,), query_id="q1")]
        summary = aggregate(results, k_values=(2,), bootstrap_samples=0)
        self.assertAlmostEqual(summary["recall_ceiling@2"], 0.5)
        self.assertLessEqual(summary["recall@2"], summary["recall_ceiling@2"])

    def test_disabling_the_bootstrap_keeps_the_same_columns(self):
        """A cheap run and a full run must stay diffable against each other."""
        self.assertEqual(
            set(self._summary(bootstrap_samples=0)),
            set(self._summary(bootstrap_samples=200)),
        )

    def test_zero_ground_truth_summary_still_carries_every_column(self):
        without_gt = evaluate_query(["a"], set(), k_values=(5,), query_id="q1")
        summary = aggregate([without_gt], k_values=(5,))
        self.assertEqual(summary["recall_ceiling@5"], 0.0)
        self.assertEqual(summary["mrr_ci_lo"], 0.0)
        self.assertEqual(summary["mrr_ci_hi"], 0.0)

    def test_means_are_unchanged_by_adding_intervals(self):
        """Item 4 must not perturb the numbers it annotates."""
        summary = self._summary(bootstrap_samples=500)
        self.assertAlmostEqual(summary["mrr"], (1.0 + 0.5 + 0.0) / 3)
        self.assertAlmostEqual(summary["recall@3"], (1.0 + 1.0 + 0.0) / 3)


if __name__ == "__main__":
    unittest.main()
