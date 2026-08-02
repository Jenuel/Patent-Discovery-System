"""Unit tests for evaluation.retrieval_eval's pure helpers
(run from apps/api: python -m unittest discover tests).

The harness's live path needs Qdrant, OpenAI and a cross-encoder, so what is
tested here is everything factored out of it: the metric-free plumbing
(dedupe, fieldnames) and the ablation configuration (EvalConfig, resolve_depth,
report_paths, config_columns). This matches FakeDense's style in
test_hierarchical.py of isolating pure logic from I/O.

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
from evaluation.retrieval_eval import (
    EvalConfig,
    Pipeline,
    config_columns,
    dedupe_patent_ids,
    report_paths,
    resolve_depth,
    union_fieldnames,
)


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

    def test_dense_retriever_accepts_the_ablation_kwargs(self):
        """build_pipeline configures the arm here, so these names must hold."""
        inspect.signature(DenseRetriever).bind(
            object(),
            arm="dense",
            fusion="dbsf",
            prefetch_multiplier=3,
            dense_prefetch_limit=60,
            sparse_prefetch_limit=20,
        )

    def test_dense_retriever_accepts_the_weighted_kwargs(self):
        inspect.signature(DenseRetriever).bind(
            object(),
            arm="weighted",
            dense_weight=0.9,
            sparse_weight=0.1,
            rrf_k=60,
        )

    def test_hierarchical_retriever_accepts_a_reranker(self):
        inspect.signature(HierarchicalRetriever).bind(
            dense=object(), cfg=HierarchicalConfig(), reranker=object()
        )


class ResolvedArmTests(unittest.TestCase):
    """Weight flags select arm="weighted" (RET-09) without a separate --arm."""

    def test_no_weights_keeps_the_requested_arm(self):
        self.assertEqual(EvalConfig().resolved_arm, "hybrid")
        self.assertEqual(EvalConfig(arm="dense").resolved_arm, "dense")
        self.assertFalse(EvalConfig(arm="dense").weighted)

    def test_either_weight_alone_selects_the_weighted_arm(self):
        """--dense-weight 1.0 is a legitimate 1:0 run, not an incomplete one."""
        self.assertEqual(EvalConfig(dense_weight=1.0).resolved_arm, "weighted")
        self.assertEqual(EvalConfig(sparse_weight=0.1).resolved_arm, "weighted")

    def test_an_explicit_weighted_arm_needs_no_flags(self):
        cfg = EvalConfig(arm="weighted")
        self.assertEqual(cfg.resolved_arm, "weighted")
        self.assertFalse(cfg.weighted)  # build_pipeline then uses the defaults

    def test_config_columns_reports_the_resolved_arm(self):
        cols = config_columns(
            EvalConfig(dense_weight=0.9, sparse_weight=0.1), reranked=False, depth=10
        )
        self.assertEqual(cols["arm"], "weighted")
        self.assertEqual(cols["fusion"], "weighted_rrf")
        self.assertEqual(cols["rrf_k"], 60)


class ResolveDepthTests(unittest.TestCase):
    """The fetch depth must cover the deepest requested k.

    dedupe_patent_ids used to cap at patent_top_k=10 regardless, so
    `--patent-k 20` reported recall@20 == recall@10 — a silently wrong number
    rather than an error.
    """

    def test_depth_is_raised_to_the_deepest_requested_k(self):
        self.assertEqual(resolve_depth(EvalConfig(), [3, 5, 20]), 20)

    def test_explicit_depth_wins_when_it_is_deeper(self):
        self.assertEqual(resolve_depth(EvalConfig(depth=200), [3, 5, 10]), 200)

    def test_never_falls_below_patent_top_k(self):
        self.assertEqual(resolve_depth(EvalConfig(patent_top_k=10), [3]), 10)


class FetchDepthTests(unittest.TestCase):
    """Stage 1 must fetch what production fetches, not what is scored.

    The hybrid arm's per-arm prefetch is a multiple of the fetch depth, so
    fetching 10 where production fetches ``dense_top_k=20`` hands Qdrant's
    fusion half the candidate pool and measures a configuration that does not
    ship. That defect moved patent MRR 0.464 -> 0.498 on the live corpus,
    larger than most effects this harness exists to detect, while every
    reported column still claimed to describe production.
    """

    def _pipeline(self, depth: int, reranker=None) -> Pipeline:
        """fetch_depth reads only the retriever, so the I/O members stay None."""
        return Pipeline(
            embedder=None,
            qdrant=None,
            retriever=HierarchicalRetriever(
                dense=DenseRetriever(None), cfg=HierarchicalConfig(), reranker=reranker
            ),
            cfg=EvalConfig(depth=depth),
            depth=depth,
        )

    def test_without_a_reranker_stage_1_fetches_dense_top_k(self):
        cfg = HierarchicalConfig()
        self.assertEqual(self._pipeline(depth=cfg.patent_top_k).fetch_depth, cfg.dense_top_k)

    def test_with_a_reranker_stage_1_fetches_rerank_candidates(self):
        cfg = HierarchicalConfig()
        pipeline = self._pipeline(depth=cfg.patent_top_k, reranker=object())
        self.assertEqual(pipeline.fetch_depth, cfg.rerank_candidates)

    def test_a_deeper_requested_depth_still_wins(self):
        """--depth 200 (Test 5's recall ceiling) must not be capped by either."""
        self.assertEqual(self._pipeline(depth=200).fetch_depth, 200)
        self.assertEqual(self._pipeline(depth=200, reranker=object()).fetch_depth, 200)

    def test_matches_the_production_stage_1_depth_exactly(self):
        """Pins fetch_depth to HierarchicalRetriever's own stage1_top_k rule."""
        cfg = HierarchicalConfig()
        for reranker in (None, object()):
            expected = cfg.rerank_candidates if reranker else cfg.dense_top_k
            with self.subTest(reranker=bool(reranker)):
                pipeline = self._pipeline(depth=cfg.patent_top_k, reranker=reranker)
                self.assertEqual(pipeline.fetch_depth, expected)


class ReportPathsTests(unittest.TestCase):
    def test_untagged_runs_keep_the_historical_filenames(self):
        per_query, summary = report_paths("results")
        self.assertEqual(per_query.name, "retrieval_eval_per_query.csv")
        self.assertEqual(summary.name, "retrieval_eval_summary.csv")

    def test_tag_gives_each_sweep_run_its_own_pair(self):
        per_query, summary = report_paths("results", "dense")
        self.assertEqual(per_query.name, "retrieval_eval_per_query__dense.csv")
        self.assertEqual(summary.name, "retrieval_eval_summary__dense.csv")

    def test_tags_that_escape_the_output_directory_are_rejected(self):
        for bad in ("../etc", "a/b", "a\\b"):
            with self.assertRaises(ValueError, msg=bad):
                report_paths("results", bad)


class EvalConfigTests(unittest.TestCase):
    def test_defaults_reproduce_production(self):
        cfg = EvalConfig()
        self.assertEqual(cfg.arm, "hybrid")
        self.assertEqual(cfg.fusion, "rrf")
        self.assertEqual(cfg.patent_top_k, HierarchicalConfig().patent_top_k)
        self.assertEqual(cfg.claim_top_k, HierarchicalConfig().claim_top_k)
        self.assertIsNone(cfg.rerank, "must defer to settings.rerank_enabled")
        self.assertFalse(cfg.weighted)

    def test_either_weight_alone_switches_on_python_fusion(self):
        self.assertTrue(EvalConfig(dense_weight=1.0).weighted)
        self.assertTrue(EvalConfig(sparse_weight=0.0).weighted)


class ConfigColumnsTests(unittest.TestCase):
    """Item 2's "no column saying so" — a CSV must record what produced it."""

    def test_rerank_state_is_recorded(self):
        cols = config_columns(
            EvalConfig(rerank_model="Xenova/ms-marco-MiniLM-L-6-v2"),
            reranked=True,
            depth=10,
        )
        self.assertTrue(cols["reranked"])
        self.assertEqual(cols["rerank_model"], "Xenova/ms-marco-MiniLM-L-6-v2")

    def test_model_is_blank_when_reranking_did_not_run(self):
        """A model name next to reranked=False would misreport the run."""
        cols = config_columns(EvalConfig(rerank_model="some-model"), reranked=False, depth=10)
        self.assertFalse(cols["reranked"])
        self.assertEqual(cols["rerank_model"], "")

    def test_weighted_fusion_is_named_as_such_not_as_rrf(self):
        cols = config_columns(EvalConfig(dense_weight=10.0, sparse_weight=1.0), False, 10)
        self.assertEqual(cols["fusion"], "weighted_rrf")
        self.assertEqual(cols["dense_weight"], 10.0)
        self.assertEqual(cols["rrf_k"], 60)

    def test_rrf_k_is_blank_under_native_fusion(self):
        """Qdrant's RRF k is fixed server-side; reporting one would be a lie."""
        self.assertEqual(config_columns(EvalConfig(), False, 10)["rrf_k"], "")


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
