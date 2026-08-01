"""Unit tests for the cross-encoder reranking stage (RET-07).

No model is downloaded and no ONNX runtime is loaded: the pure helpers are
exercised directly, and the class-level tests inject a FakeReranker or stub the
scoring call. This mirrors test_qdrant_store.py, which keeps the BM25 encoder
out of the suite the same way.

Run from apps/api: python -m unittest discover tests
"""
from __future__ import annotations

import asyncio
import unittest
from typing import Any, Dict, List, Optional, Sequence

from app.services.rerank.reranker import (
    CrossEncoderReranker,
    RerankConfig,
    apply_scores,
    document_text,
)
from app.services.retrieval.fusion import ScoredMatch
from app.services.retrieval.hierarchical import HierarchicalConfig, HierarchicalRetriever


def _match(id_: str, score: float = 1.0, **metadata: Any) -> ScoredMatch:
    return ScoredMatch(id=id_, score=score, metadata=metadata)


def _run(coro):
    return asyncio.run(coro)


class DocumentTextTests(unittest.TestCase):
    def test_joins_title_and_abstract(self):
        m = _match("p1", title="Systolic Array", abstract="A hardware accelerator.")
        self.assertEqual(document_text(m, 1000), "Systolic Array A hardware accelerator.")

    def test_falls_back_to_text_for_claim_level_matches(self):
        m = _match("c1", text="A method comprising the steps of.")
        self.assertEqual(document_text(m, 1000), "A method comprising the steps of.")

    def test_truncates_to_budget(self):
        m = _match("p1", abstract="x" * 5000)
        self.assertEqual(len(document_text(m, 1200)), 1200)

    def test_missing_fields_yield_empty_string(self):
        self.assertEqual(document_text(_match("p1"), 1000), "")
        self.assertEqual(document_text(_match("p1", title=None, abstract=""), 1000), "")

    def test_tolerates_a_match_with_no_metadata(self):
        self.assertEqual(document_text(ScoredMatch(id="p1", score=1.0, metadata={}), 1000), "")


class ApplyScoresTests(unittest.TestCase):
    def test_reorders_descending_by_score(self):
        ms = [_match("a"), _match("b"), _match("c")]
        out = apply_scores(ms, [0.1, 0.9, 0.5], top_n=3)
        self.assertEqual([m.id for m in out], ["b", "c", "a"])

    def test_truncates_to_top_n(self):
        ms = [_match("a"), _match("b"), _match("c")]
        self.assertEqual([m.id for m in apply_scores(ms, [0.1, 0.9, 0.5], top_n=2)], ["b", "c"])

    def test_promotes_a_candidate_from_beyond_the_original_cutoff(self):
        """The whole point of RET-07: rank-11+ documents becoming reachable."""
        ms = [_match(f"p{i}") for i in range(12)]
        scores = [0.0] * 11 + [9.9]          # p11 was last, is now best
        self.assertEqual(apply_scores(ms, scores, top_n=1)[0].id, "p11")

    def test_length_mismatch_raises(self):
        """Silently mis-pairing documents with scores would corrupt ranking."""
        with self.assertRaises(ValueError):
            apply_scores([_match("a"), _match("b")], [1.0], top_n=2)

    def test_non_positive_top_n_raises(self):
        with self.assertRaises(ValueError):
            apply_scores([_match("a")], [1.0], top_n=0)

    def test_empty_input(self):
        self.assertEqual(apply_scores([], [], top_n=5), [])


class _StubbedReranker(CrossEncoderReranker):
    """CrossEncoderReranker with the model call replaced — no ONNX, no download."""

    def __init__(self, scores: Optional[List[float]] = None, raises: bool = False):
        super().__init__(RerankConfig(max_document_chars=1200))
        self._scores = scores
        self._raises = raises
        self.seen_documents: Optional[List[str]] = None

    def _score(self, query: str, documents: List[str]) -> List[float]:
        if self._raises:
            raise RuntimeError("model exploded")
        self.seen_documents = documents
        return self._scores if self._scores is not None else [0.0] * len(documents)


class RerankTests(unittest.TestCase):
    def test_reorders_by_cross_encoder_score(self):
        ms = [_match("a", title="A"), _match("b", title="B"), _match("c", title="C")]
        r = _StubbedReranker(scores=[0.1, 0.9, 0.5])
        self.assertEqual([m.id for m in _run(r.rerank("q", ms, top_n=3))], ["b", "c", "a"])

    def test_model_failure_falls_back_to_retrieval_order(self):
        """A degraded ranking is acceptable; an exception to the caller is not."""
        ms = [_match("a", title="A"), _match("b", title="B")]
        out = _run(_StubbedReranker(raises=True).rerank("q", ms, top_n=2))
        self.assertEqual([m.id for m in out], ["a", "b"])

    def test_textless_candidates_are_kept_but_ranked_last(self):
        ms = [_match("a"), _match("b", title="B"), _match("c", title="C")]
        r = _StubbedReranker(scores=[0.2, 0.8])       # only b and c are scoreable
        out = _run(r.rerank("q", ms, top_n=3))
        self.assertEqual([m.id for m in out], ["c", "b", "a"])

    def test_all_textless_keeps_original_order(self):
        ms = [_match("a"), _match("b")]
        out = _run(_StubbedReranker().rerank("q", ms, top_n=2))
        self.assertEqual([m.id for m in out], ["a", "b"])

    def test_only_scoreable_documents_reach_the_model(self):
        ms = [_match("a"), _match("b", title="Real Title")]
        r = _StubbedReranker(scores=[1.0])
        _run(r.rerank("q", ms, top_n=2))
        self.assertEqual(r.seen_documents, ["Real Title"])

    def test_empty_and_single_candidate_shortcuts(self):
        self.assertEqual(_run(_StubbedReranker().rerank("q", [], top_n=5)), [])
        one = [_match("a", title="A")]
        self.assertEqual([m.id for m in _run(_StubbedReranker().rerank("q", one, top_n=5))], ["a"])

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            _run(_StubbedReranker().rerank("", [_match("a", title="A")], top_n=1))


# ----------------------------------------------------------------------
# Integration with HierarchicalRetriever
# ----------------------------------------------------------------------

class FakeDense:
    """Records the top_k it was asked for, so candidate widening is observable."""

    def __init__(self, patent_hits: int = 50):
        self.patent_hits = patent_hits
        self.calls: List[Dict[str, Any]] = []

    async def search(self, *, dense_vector, top_k, metadata_filter, level, query_text=None):
        self.calls.append({"level": level, "top_k": top_k, "query_text": query_text,
                           "metadata_filter": metadata_filter})
        if level == "patent":
            return [
                {"id": f"p{i}", "score": 1.0 - i / 100,
                 "metadata": {"patent_id": f"US{i}", "title": f"Title {i}"}}
                for i in range(min(top_k, self.patent_hits))
            ]
        return [{"id": "c1", "score": 0.9, "metadata": {"patent_id": "US0"}}]


class FakeReranker:
    """Reverses the candidate list — makes promotion from the tail unmistakable."""

    def __init__(self):
        self.called_with: Optional[Dict[str, Any]] = None

    async def rerank(self, query, matches, top_n):
        self.called_with = {"query": query, "n_candidates": len(matches), "top_n": top_n}
        return list(reversed(list(matches)))[:top_n]


class HierarchicalRerankIntegrationTests(unittest.TestCase):
    def test_without_reranker_behaviour_is_unchanged(self):
        dense = FakeDense()
        r = HierarchicalRetriever(dense=dense, cfg=HierarchicalConfig())
        _run(r.retrieve_claims_hierarchical(
            dense_query_vec=[0.1], query_text="q", base_filter={}))
        self.assertEqual(dense.calls[0]["top_k"], HierarchicalConfig().dense_top_k)

    def test_with_reranker_stage1_fetches_the_wider_candidate_list(self):
        dense = FakeDense()
        cfg = HierarchicalConfig()
        r = HierarchicalRetriever(dense=dense, cfg=cfg, reranker=FakeReranker())
        _run(r.retrieve_claims_hierarchical(
            dense_query_vec=[0.1], query_text="q", base_filter={}))
        self.assertEqual(dense.calls[0]["top_k"], cfg.rerank_candidates)
        self.assertGreater(cfg.rerank_candidates, cfg.dense_top_k)

    def test_reranker_receives_the_candidates_and_the_query(self):
        dense, rr = FakeDense(), FakeReranker()
        cfg = HierarchicalConfig()
        r = HierarchicalRetriever(dense=dense, cfg=cfg, reranker=rr)
        _run(r.retrieve_claims_hierarchical(
            dense_query_vec=[0.1], query_text="hardware accelerator", base_filter={}))
        self.assertEqual(rr.called_with["query"], "hardware accelerator")
        self.assertEqual(rr.called_with["n_candidates"], cfg.rerank_candidates)
        self.assertEqual(rr.called_with["top_n"], cfg.patent_top_k)

    def test_stage2_filters_on_the_reranked_patent_ids(self):
        """
        The load-bearing assertion: reranking must actually change which
        patents Stage 2 searches. FakeReranker reverses the candidate list, so
        Stage 2 should see the TAIL of the 50 candidates (US49..US40), not the
        head (US0..US9) that plain retrieval would have selected.
        """
        dense, cfg = FakeDense(), HierarchicalConfig()
        r = HierarchicalRetriever(dense=dense, cfg=cfg, reranker=FakeReranker())
        _run(r.retrieve_claims_hierarchical(
            dense_query_vec=[0.1], query_text="q", base_filter={}))

        claim_call = dense.calls[1]
        self.assertEqual(claim_call["level"], "claim")
        selected = claim_call["metadata_filter"]["patent_id"]["$in"]

        # Derived from cfg, not hardcoded: patent_top_k is a tuning knob
        # (RET-08 moved it 10 -> 20) and this test asserts ordering, not sizing.
        n = cfg.patent_top_k
        expected = [f"US{i}" for i in range(cfg.rerank_candidates - 1,
                                            cfg.rerank_candidates - 1 - n, -1)]
        self.assertEqual(selected, expected)
        self.assertNotIn("US0", selected, "Stage 2 still used the pre-rerank head")

    def test_rerank_is_skipped_without_query_text(self):
        """A cross-encoder has nothing to score against a bare vector."""
        dense, rr = FakeDense(), FakeReranker()
        r = HierarchicalRetriever(dense=dense, cfg=HierarchicalConfig(), reranker=rr)
        _run(r.retrieve_claims_hierarchical(
            dense_query_vec=[0.1], query_text=None, base_filter={}))
        self.assertIsNone(rr.called_with)

    def test_reranker_failure_does_not_break_retrieval(self):
        class Exploding:
            async def rerank(self, query, matches, top_n):
                raise RuntimeError("boom")

        dense = FakeDense()
        r = HierarchicalRetriever(dense=dense, cfg=HierarchicalConfig(), reranker=Exploding())
        out = _run(r.retrieve_claims_hierarchical(
            dense_query_vec=[0.1], query_text="q", base_filter={}))
        self.assertEqual(dense.calls[1]["level"], "claim")   # stage 2 still ran
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main()
