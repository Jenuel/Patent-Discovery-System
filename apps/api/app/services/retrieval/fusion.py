from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_RRF_K = 60


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
    Convert Qdrant query results to ScoredMatch objects.
    """
    return [
        ScoredMatch(
            id=r.get("id", ""),
            score=r.get("score", 0.0),
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]


def to_result_dicts(matches: Sequence[ScoredMatch]) -> List[Dict[str, Any]]:
    """
    Inverse of :func:`to_scored_matches`.

    Client-side fusion works in ``ScoredMatch`` space, but every
    ``DenseRetriever.search`` arm must return the ``[{id, score, metadata}]``
    dict schema its callers expect — ``HierarchicalRetriever`` calls
    ``to_scored_matches`` on whatever comes back.
    """
    return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in matches]


def weighted_rrf(
    arms: Sequence[Tuple[float, Sequence[ScoredMatch]]],
    k: int = DEFAULT_RRF_K,
    limit: Optional[int] = None,
) -> List[ScoredMatch]:
    """Weighted Reciprocal Rank Fusion over independently-retrieved arms.

    ``score(d) = Σ_arm weight_arm / (k + rank_arm(d))``, rank starting at 1.

    Client-side because Qdrant's native ``FusionQuery`` has no weight
    parameter (``qdrant.py``): the only way to vary the balance between the
    dense and BM25 arms is to fetch each separately and fuse here. That is what
    ``DenseRetriever(arm="weighted")`` does, and it is the reason that arm costs
    two round trips where the native hybrid arm costs one.
    """
    scores: Dict[str, float] = {}
    first_seen: Dict[str, ScoredMatch] = {}

    for weight, matches in arms:
        if not weight:
            continue
        for rank, match in enumerate(matches, start=1):
            scores[match.id] = scores.get(match.id, 0.0) + weight / (k + rank)
            first_seen.setdefault(match.id, match)

    order = sorted(first_seen, key=lambda mid: -scores[mid])
    fused = [replace(first_seen[mid], score=scores[mid]) for mid in order]
    return fused[:limit] if limit is not None else fused