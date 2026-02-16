from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set


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
    Convert Pinecone/Elasticsearch query results to ScoredMatch objects.
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
    Reciprocal Rank Fusion (RRF) with sparse retrieval as source of truth.
    Only patent IDs present in sparse results are retained.
    
    Args:
        dense_results: Results from dense/semantic retrieval (Pinecone)
        sparse_results: Results from sparse/lexical retrieval (Elasticsearch)
        k: RRF constant (default 60)
        top_k: Number of results to return
    
    Returns:
        Fused and re-ranked results, filtered to sparse result patent IDs
    """
    # Create allowlist of patent IDs from sparse results
    sparse_patent_ids: Set[str] = {
        match.metadata.get("patent_id", match.id) 
        for match in sparse_results
    }
    
    scores: Dict[str, float] = {}
    metadata_map: Dict[str, Dict[str, Any]] = {}
    
    # Process dense results (only if patent_id in sparse allowlist)
    for rank, match in enumerate(dense_results, 1):
        patent_id = match.metadata.get("patent_id", match.id)
        
        if patent_id in sparse_patent_ids:
            scores[patent_id] = scores.get(patent_id, 0.0) + 1.0 / (k + rank)
            metadata_map[patent_id] = match.metadata
    
    # Process sparse results (all included)
    for rank, match in enumerate(sparse_results, 1):
        patent_id = match.metadata.get("patent_id", match.id)
        
        scores[patent_id] = scores.get(patent_id, 0.0) + 1.0 / (k + rank)
        # Prefer sparse metadata if not already set
        if patent_id not in metadata_map:
            metadata_map[patent_id] = match.metadata
    
    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Build result list
    results = [
        ScoredMatch(
            id=patent_id,
            score=scores[patent_id],
            metadata=metadata_map[patent_id],
        )
        for patent_id in sorted_ids[:top_k]
    ]
    
    return results