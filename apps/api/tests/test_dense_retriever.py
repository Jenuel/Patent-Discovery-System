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


class FakeStore:
    """Records which search method was called, and with what."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def search_hybrid(self, **kwargs):
        self.calls.append({"method": "search_hybrid", **kwargs})
        return []

    async def search_patents_dense(self, **kwargs):
        self.calls.append({"method": "search_patents_dense", **kwargs})
        return []

    async def search_bm25(self, **kwargs):
        self.calls.append({"method": "search_bm25", **kwargs})
        return []

    async def search_claims_dense(self, **kwargs):
        self.calls.append({"method": "search_claims_dense", **kwargs})
        return []


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


if __name__ == "__main__":
    unittest.main()
