from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagPolicy:
    """RAG policy configuration for final result selection.
    
    The final_top_n represents the number of evidence items to show to the user
    after all retrieval, fusion, and ranking stages are complete.
    
    For a dataset with 113 patents and 2,200 claims, returning 8 final results
    provides a focused, high-quality set of evidence without overwhelming the user.
    """
    final_top_n: int = 8


DEFAULT_POLICY = RagPolicy(final_top_n=8)
