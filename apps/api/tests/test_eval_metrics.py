"""Unit tests for evaluation.metrics (run from apps/api: python -m unittest discover tests)."""
from __future__ import annotations

import unittest

from evaluation.metrics import (
    aggregate,
    dcg_at_k,
    evaluate_query,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
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


if __name__ == "__main__":
    unittest.main()
