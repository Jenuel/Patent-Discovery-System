from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class ScoredMatch:
    """
    Unified result format for retrieval fusion.
    """
    id: str
    score: float
    metadata: Dict[str, Any]


def to_scored_matches(results: Sequence[Dict[str, Any]]) -> List[ScoredMatch]:
    """
    Convert Pinecone query results to ScoredMatch objects.
    """
    return [
        ScoredMatch(
            id=r.get("id", ""),
            score=r.get("score", 0.0),
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]


def fuse_rrf(
    dense_results: Sequence[ScoredMatch],
    sparse_results: Sequence[ScoredMatch],
    k: int = 60,
    top_k: int = 20,
) -> List[ScoredMatch]:
    """
    Reciprocal Rank Fusion (RRF) to combine dense and sparse retrieval.
    
    Args:
        dense_results: Results from dense/semantic retrieval
        sparse_results: Results from sparse/lexical retrieval
        k: RRF constant (default 60)
        top_k: Number of results to return
    
    Returns:
        Fused and re-ranked results
    """
    scores: Dict[str, float] = {}
    metadata_map: Dict[str, Dict[str, Any]] = {}
    
    # Process dense results
    for rank, match in enumerate(dense_results, 1):
        scores[match.id] = scores.get(match.id, 0.0) + 1.0 / (k + rank)
        metadata_map[match.id] = match.metadata
    
    # Process sparse results
    for rank, match in enumerate(sparse_results, 1):
        scores[match.id] = scores.get(match.id, 0.0) + 1.0 / (k + rank)
        if match.id not in metadata_map:
            metadata_map[match.id] = match.metadata
    
    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Build result list
    results = [
        ScoredMatch(
            id=doc_id,
            score=scores[doc_id],
            metadata=metadata_map.get(doc_id, {}),
        )
        for doc_id in sorted_ids[:top_k]
    ]
    
    return results
