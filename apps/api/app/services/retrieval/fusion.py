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
    Convert Pinecone/Qdrant query results to ScoredMatch objects.
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
    Standard Reciprocal Rank Fusion (RRF) over the union of both sources.

    Every patent ID from either source is scored; IDs found by both sources
    accumulate score from each and rank higher. If one source returns nothing
    (e.g. sparse retrieval is empty or skipped), the fusion degrades to the
    other source's ranking instead of discarding results.

    Args:
        dense_results: Results from dense/semantic retrieval (Pinecone)
        sparse_results: Results from sparse/lexical retrieval (Qdrant BM25)
        k: RRF constant (default 60)
        top_k: Number of results to return

    Returns:
        Fused and re-ranked results from the union of both sources
    """
    scores: Dict[str, float] = {}
    metadata_map: Dict[str, Dict[str, Any]] = {}

    for source in (dense_results, sparse_results):
        for rank, match in enumerate(source, 1):
            patent_id = match.metadata.get("patent_id", match.id)

            scores[patent_id] = scores.get(patent_id, 0.0) + 1.0 / (k + rank)
            # First source to see an ID provides its metadata (dense preferred)
            if patent_id not in metadata_map:
                metadata_map[patent_id] = match.metadata

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [
        ScoredMatch(
            id=patent_id,
            score=scores[patent_id],
            metadata=metadata_map[patent_id],
        )
        for patent_id in sorted_ids[:top_k]
    ]