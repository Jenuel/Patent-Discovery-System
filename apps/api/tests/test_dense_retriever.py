"""Unit tests for DenseRetriever's patent-arm routing.

The arm and its fusion parameters are set at construction so that
``search()``'s signature stays fixed for ``HierarchicalRetriever`` while the
evaluation harness can still ablate arms (EVAL_ABLATION.md Tests 1, 3). These
tests pin that routing against a fake store — no Qdrant, no fastembed.

Run from apps/api: python -m unittest discover tests
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List

from app.services.retrieval.dense import PATENT_ARMS, DenseRetriever


def _hits(*ids):
    """Result dicts in the store's `[{id, score, metadata}]` schema."""
    return [
        {"id": i, "score": 1.0, "metadata": {"patent_id": i}} for i in ids
    ]


class FakeStore:
    """Records which search method was called, and with what.

    ``dense_results``/``sparse_results`` let the weighted-arm tests control what
    each arm returns; they default to empty, which is what the arm-routing tests
    want.
    """

    def __init__(self, dense_results=None, sparse_results=None):
        self.calls: List[Dict[str, Any]] = []
        self.dense_results = dense_results or []
        self.sparse_results = sparse_results or []

    async def search_hybrid(self, **kwargs):
        self.calls.append({"method": "search_hybrid", **kwargs})
        return []

    async def search_patents_dense(self, **kwargs):
        self.calls.append({"method": "search_patents_dense", **kwargs})
        return self.dense_results

    async def search_bm25(self, **kwargs):
        self.calls.append({"method": "search_bm25", **kwargs})
        return self.sparse_results

    async def search_claims_dense(self, **kwargs):
        self.calls.append({"method": "search_claims_dense", **kwargs})
        return []

    def method_names(self):
        return {c["method"] for c in self.calls}

    def call_to(self, method):
        return next(c for c in self.calls if c["method"] == method)


async def _search(retriever: DenseRetriever, **overrides) -> Dict[str, Any]:
    kwargs = {
        "dense_vector": [0.1, 0.2],
        "top_k": 5,
        "metadata_filter": {},
        "level": "patent",
        "query_text": "battery cathode",
    }
    kwargs.update(overrides)
    await retriever.search(**kwargs)
    return retriever.store.calls[-1]


class ArmRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_is_the_default_arm(self):
        store = FakeStore()
        call = await _search(DenseRetriever(store))
        self.assertEqual(call["method"], "search_hybrid")

    async def test_dense_arm_skips_the_bm25_arm_entirely(self):
        store = FakeStore()
        call = await _search(DenseRetriever(store, arm="dense"))
        self.assertEqual(call["method"], "search_patents_dense")

    async def test_dense_arm_ignores_query_text(self):
        """The point of --arm dense is that BM25 never runs, text or no text."""
        store = FakeStore()
        call = await _search(DenseRetriever(store, arm="dense"), query_text="a title")
        self.assertEqual(call["method"], "search_patents_dense")

    async def test_bm25_arm_routes_to_sparse_search(self):
        store = FakeStore()
        call = await _search(DenseRetriever(store, arm="bm25"))
        self.assertEqual(call["method"], "search_bm25")
        self.assertEqual(call["query_text"], "battery cathode")

    async def test_bm25_arm_without_query_text_raises(self):
        """Unlike hybrid, the sparse arm has no dense vector to fall back to."""
        with self.assertRaises(ValueError):
            await _search(DenseRetriever(FakeStore(), arm="bm25"), query_text=None)

    async def test_hybrid_falls_back_to_dense_without_query_text(self):
        store = FakeStore()
        call = await _search(DenseRetriever(store), query_text=None)
        self.assertEqual(call["method"], "search_patents_dense")

    async def test_claim_level_is_unaffected_by_the_arm(self):
        """claims_hybrid has no sparse vectors, so there is no arm to pick."""
        for arm in PATENT_ARMS:
            store = FakeStore()
            call = await _search(DenseRetriever(store, arm=arm), level="claim")
            self.assertEqual(call["method"], "search_claims_dense", f"arm={arm}")

    def test_unknown_arm_raises_at_construction(self):
        with self.assertRaises(ValueError):
            DenseRetriever(FakeStore(), arm="sparse")


class FusionParameterForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_match_production(self):
        store = FakeStore()
        call = await _search(DenseRetriever(store))
        self.assertEqual(call["fusion"], "rrf")
        self.assertEqual(call["prefetch_multiplier"], 3)
        self.assertIsNone(call["dense_prefetch_limit"])
        self.assertIsNone(call["sparse_prefetch_limit"])

    async def test_ablation_parameters_reach_the_store(self):
        """EVAL_ABLATION.md Test 3 sweeps sparse prefetch with dense held fixed."""
        store = FakeStore()
        retriever = DenseRetriever(
            store, fusion="dbsf", dense_prefetch_limit=60, sparse_prefetch_limit=20
        )
        call = await _search(retriever)
        self.assertEqual(call["fusion"], "dbsf")
        self.assertEqual(call["dense_prefetch_limit"], 60)
        self.assertEqual(call["sparse_prefetch_limit"], 20)


class WeightedArmTests(unittest.IsolatedAsyncioTestCase):
    """RET-09 — client-side weighted RRF (EVAL_ABLATION.md Test 6).

    Qdrant's native FusionQuery takes no weights, so this arm fetches both
    arms itself and fuses in Python. These tests pin that it really does query
    both, at the native candidate depth, and that the weights reach the fusion.
    """

    async def _run(self, retriever, **overrides):
        await _search(retriever, **overrides)
        return retriever.store

    async def test_weighted_arm_queries_both_arms_not_search_hybrid(self):
        store = FakeStore()
        await self._run(DenseRetriever(store, arm="weighted"))
        self.assertEqual(
            store.method_names(), {"search_patents_dense", "search_bm25"}
        )

    async def test_both_arms_fetch_at_the_native_prefetch_depth(self):
        """Fusing over a shallower pool than search_hybrid would confound any
        weighted-vs-native comparison with depth."""
        store = FakeStore()
        await self._run(
            DenseRetriever(store, arm="weighted", prefetch_multiplier=3), top_k=20
        )
        self.assertEqual(store.call_to("search_patents_dense")["top_k"], 60)
        self.assertEqual(store.call_to("search_bm25")["top_k"], 60)

    async def test_absolute_prefetch_overrides_are_honoured_per_arm(self):
        store = FakeStore()
        await self._run(
            DenseRetriever(
                store,
                arm="weighted",
                dense_prefetch_limit=60,
                sparse_prefetch_limit=20,
            ),
            top_k=10,
        )
        self.assertEqual(store.call_to("search_patents_dense")["top_k"], 60)
        self.assertEqual(store.call_to("search_bm25")["top_k"], 20)

    async def test_results_are_fused_and_truncated_to_top_k(self):
        store = FakeStore(dense_results=_hits("a", "b", "c"), sparse_results=_hits("z"))
        retriever = DenseRetriever(store, arm="weighted")
        out = await retriever.search(
            dense_vector=[0.1],
            top_k=2,
            metadata_filter={},
            level="patent",
            query_text="q",
        )
        self.assertEqual([r["id"] for r in out], ["a", "b"])
        self.assertEqual(out[0]["metadata"], {"patent_id": "a"})

    async def test_the_result_schema_matches_every_other_arm(self):
        """HierarchicalRetriever calls to_scored_matches on whatever comes back."""
        store = FakeStore(dense_results=_hits("a"))
        retriever = DenseRetriever(store, arm="weighted")
        out = await retriever.search(
            dense_vector=[0.1],
            top_k=5,
            metadata_filter={},
            level="patent",
            query_text="q",
        )
        self.assertEqual(set(out[0]), {"id", "score", "metadata"})

    async def test_weights_change_the_ordering(self):
        dense, sparse = _hits("d1", "d2"), _hits("s1", "s2")

        async def order(dense_weight, sparse_weight):
            store = FakeStore(dense_results=dense, sparse_results=sparse)
            retriever = DenseRetriever(
                store,
                arm="weighted",
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                rrf_k=1,
            )
            out = await retriever.search(
                dense_vector=[0.1],
                top_k=4,
                metadata_filter={},
                level="patent",
                query_text="q",
            )
            return [r["id"] for r in out]

        self.assertEqual((await order(0.9, 0.1))[:2], ["d1", "d2"])
        self.assertEqual((await order(0.1, 0.9))[:2], ["s1", "s2"])

    async def test_an_empty_bm25_arm_degrades_to_dense_ordering(self):
        """Term-less queries produce no sparse hits (RET-06)."""
        store = FakeStore(dense_results=_hits("a", "b"), sparse_results=[])
        retriever = DenseRetriever(store, arm="weighted")
        out = await retriever.search(
            dense_vector=[0.1],
            top_k=5,
            metadata_filter={},
            level="patent",
            query_text="q",
        )
        self.assertEqual([r["id"] for r in out], ["a", "b"])

    async def test_falls_back_to_dense_only_without_query_text(self):
        """Same contract as the hybrid arm — BM25 has nothing to match on."""
        store = FakeStore()
        await self._run(DenseRetriever(store, arm="weighted"), query_text=None)
        self.assertEqual(store.method_names(), {"search_patents_dense"})

    async def test_claim_level_is_unaffected(self):
        store = FakeStore()
        await self._run(DenseRetriever(store, arm="weighted"), level="claim")
        self.assertEqual(store.method_names(), {"search_claims_dense"})

    async def test_the_metadata_filter_reaches_both_arms(self):
        store = FakeStore()
        await self._run(
            DenseRetriever(store, arm="weighted"),
            metadata_filter={"filing_year": 2018},
        )
        for method in ("search_patents_dense", "search_bm25"):
            self.assertEqual(
                store.call_to(method)["metadata_filter"], {"filing_year": 2018}, method
            )

    def test_default_weights_are_the_ratio_test_6_identified(self):
        retriever = DenseRetriever(FakeStore(), arm="weighted")
        self.assertEqual((retriever.dense_weight, retriever.sparse_weight), (0.9, 0.1))

    def test_both_weights_zero_raises_at_construction(self):
        """Fusing nothing would return an empty list on every query instead."""
        with self.assertRaises(ValueError):
            DenseRetriever(FakeStore(), arm="weighted", dense_weight=0.0, sparse_weight=0.0)

    def test_zero_weights_are_fine_on_the_arms_that_ignore_them(self):
        DenseRetriever(FakeStore(), arm="hybrid", dense_weight=0.0, sparse_weight=0.0)

    def test_a_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            DenseRetriever(FakeStore(), arm="weighted", sparse_weight=-0.1)


if __name__ == "__main__":
    unittest.main()
