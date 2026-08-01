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