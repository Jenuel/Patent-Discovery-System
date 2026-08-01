from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagPolicy:
    """RAG policy configuration for final result selection.

    The final_top_n represents the number of evidence items to show to the user
    after all retrieval, fusion, and ranking stages are complete. It also caps
    what reaches the LLM, since the slice happens before answer generation.

    8 was chosen for a 113-patent / 2,200-chunk corpus. Against the current
    6,000-patent corpus that cut too deep: EVAL_BASELINE.md measures claim-level
    hit_rate at 0.5263 (k=5) and 0.7368 (k=10) but 1.0000 (k=20), so a slice of
    8 hid relevant claims that Stage 2 had already retrieved. All three query
    modes are coverage-oriented (see EVAL_RERANKING.md), so 20 is the better
    trade despite precision@20 falling to 0.1000.

    Bounded by HierarchicalConfig.claim_top_k (30) — nothing beyond that exists
    to slice.
    """
    final_top_n: int = 20


DEFAULT_POLICY = RagPolicy(final_top_n=20)
