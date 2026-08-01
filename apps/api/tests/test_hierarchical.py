"""Unit tests for HierarchicalRetriever (run from apps/api: python -m unittest discover tests)."""
from __future__ import annotations

import unittest
from typing import Any, Dict, List

from app.services.retrieval.hierarchical import HierarchicalConfig, HierarchicalRetriever


class FakeDense:
    """Records every call so tests can assert on the filters/levels used."""

    def __init__(self, patent_results: List[Dict[str, Any]], claim_results: List[Dict[str, Any]]):
        self.calls: List[Dict[str, Any]] = []
        self._patent_results = patent_results
        self._claim_results = claim_results

    async def search(self, dense_vector, top_k, metadata_filter, level, query_text=None):
        self.calls.append(
            {
                "level": level,
                "metadata_filter": metadata_filter,
                "top_k": top_k,
                "query_text": query_text,
            }
        )
        return self._patent_results if level == "patent" else self._claim_results


def _dense_patent(patent_id: str) -> Dict[str, Any]:
    return {"id": patent_id, "score": 0.9, "metadata": {"patent_id": patent_id}}


class ClaimFilterScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage1_patent_allowlist_cannot_be_overridden_by_base_filter(self):
        dense = FakeDense(
            patent_results=[_dense_patent("US1")],
            claim_results=[],
        )
        retriever = HierarchicalRetriever(dense=dense, cfg=HierarchicalConfig())

        # A caller-supplied filter that (accidentally or not) also sets patent_id.
        base_filter = {"patent_id": {"$eq": "SHOULD_NOT_WIN"}, "year": {"$gte": 2020}}

        await retriever.retrieve_claims_hierarchical(
            dense_query_vec=[0.1, 0.2],
            query_text="battery cathode",
            base_filter=base_filter,
        )

        claim_filter = dense.calls[1]["metadata_filter"]
        self.assertEqual(claim_filter["patent_id"], {"$in": ["US1"]})

    async def test_patent_attribute_filters_are_applied_at_stage1_only(self):
        """base_filter describes patents, so it must not reach the claim collection.

        Claim payloads come from the Mongo chunk docs (`filing_year`, `cpc`);
        forwarding the API's `year` filter here would match zero claims and
        silently empty the pipeline.
        """
        dense = FakeDense(
            patent_results=[_dense_patent("US1")],
            claim_results=[{"id": "claim-1", "score": 0.5, "metadata": {"patent_id": "US1"}}],
        )
        retriever = HierarchicalRetriever(dense=dense, cfg=HierarchicalConfig())

        base_filter = {"year": {"$gte": 2020}, "cpc": {"$in": ["G06N"]}}

        result = await retriever.retrieve_claims_hierarchical(
            dense_query_vec=[0.1, 0.2],
            query_text="battery cathode",
            base_filter=base_filter,
        )

        stage1_filter = dense.calls[0]["metadata_filter"]
        stage2_filter = dense.calls[1]["metadata_filter"]

        # Stage 1 enforces the selection...
        self.assertEqual(stage1_filter["year"], {"$gte": 2020})
        self.assertEqual(stage1_filter["cpc"], {"$in": ["G06N"]})
        # ...and Stage 2 narrows by patent_id alone.
        self.assertEqual(stage2_filter, {"patent_id": {"$in": ["US1"]}})
        self.assertEqual([m.id for m in result], ["claim-1"])

    async def test_base_filter_is_not_mutated(self):
        dense = FakeDense(
            patent_results=[_dense_patent("US1")],
            claim_results=[],
        )
        retriever = HierarchicalRetriever(dense=dense, cfg=HierarchicalConfig())

        base_filter = {"year": {"$gte": 2020}}
        await retriever.retrieve_claims_hierarchical(
            dense_query_vec=[0.1, 0.2],
            query_text="battery cathode",
            base_filter=base_filter,
        )

        self.assertEqual(base_filter, {"year": {"$gte": 2020}})


if __name__ == "__main__":
    unittest.main()
