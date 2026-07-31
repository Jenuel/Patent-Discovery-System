"""Pure IR metric functions for retrieval evaluation.

No imports beyond the standard library — fully offline-testable with plain
lists/sets, no fakes needed.

Conventions:
  - A query whose ground-truth relevant set is empty is a valid "no correct
    answer exists" case. ``evaluate_query`` marks it ``has_ground_truth=False``
    and still returns a full ``QueryResult`` (metrics 0.0) so per-query
    reporting stays inspectable; ``aggregate`` excludes such queries from its
    means rather than scoring them as 0 or 1, either of which would distort
    the average.
  - ``precision_at_k`` divides by ``min(k, len(retrieved_ids))``, not by
    ``k`` — a retriever that legitimately returns fewer than ``k`` candidates
    isn't penalized as if the shortfall were irrelevant results.
  - ``dcg_at_k``/``ndcg_at_k`` use the linear-gain formula
    ``sum(rel_i / log2(rank + 1))`` (rank starting at 1), not exponential
    gain — simpler to hand-verify given grades are only ``{1, 2}`` in the
    example corpus. ``ndcg_at_k`` divides by the ideal DCG (grades sorted
    descending, truncated to ``k``); if the ideal DCG is 0, it returns 0.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import AbstractSet, Dict, Mapping, Optional, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    denom = min(k, len(retrieved_ids))
    if denom == 0:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / denom


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: AbstractSet[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: Sequence[str], relevance_grades: Mapping[str, float], k: int) -> float:
    total = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        grade = relevance_grades.get(rid, 0.0)
        if grade:
            total += grade / math.log2(rank + 1)
    return total


def ndcg_at_k(retrieved_ids: Sequence[str], relevance_grades: Mapping[str, float], k: int) -> float:
    actual = dcg_at_k(retrieved_ids, relevance_grades, k)
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1) if grade)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    recall: float
    precision: float
    hit_rate: float
    ndcg: float


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    has_ground_truth: bool
    reciprocal_rank: float
    num_relevant: int
    num_retrieved: int
    metrics_by_k: Dict[int, RetrievalMetrics] = field(default_factory=dict)


def evaluate_query(
    retrieved_ids: Sequence[str],
    relevant_ids: AbstractSet[str],
    k_values: Sequence[int] = (5, 10, 20),
    relevance_grades: Optional[Mapping[str, float]] = None,
    query_id: str = "",
) -> QueryResult:
    has_ground_truth = bool(relevant_ids)
    grades = relevance_grades if relevance_grades is not None else {rid: 1.0 for rid in relevant_ids}

    metrics_by_k: Dict[int, RetrievalMetrics] = {}
    for k in k_values:
        metrics_by_k[k] = RetrievalMetrics(
            k=k,
            recall=recall_at_k(retrieved_ids, relevant_ids, k) if has_ground_truth else 0.0,
            precision=precision_at_k(retrieved_ids, relevant_ids, k) if has_ground_truth else 0.0,
            hit_rate=hit_rate_at_k(retrieved_ids, relevant_ids, k) if has_ground_truth else 0.0,
            ndcg=ndcg_at_k(retrieved_ids, grades, k) if has_ground_truth else 0.0,
        )

    return QueryResult(
        query_id=query_id,
        has_ground_truth=has_ground_truth,
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids) if has_ground_truth else 0.0,
        num_relevant=len(relevant_ids),
        num_retrieved=len(retrieved_ids),
        metrics_by_k=metrics_by_k,
    )


def aggregate(results: Sequence[QueryResult], k_values: Sequence[int]) -> Dict[str, float]:
    """Mean MRR + mean recall/precision/hit_rate/ndcg per k, averaged only
    over queries where has_ground_truth is True. Keys: "mrr", "recall@5",
    "precision@5", "ndcg@5", "hit_rate@5", ..., "num_queries",
    "num_queries_with_ground_truth".
    """
    scored = [r for r in results if r.has_ground_truth]
    n = len(scored)

    out: Dict[str, float] = {
        "num_queries": float(len(results)),
        "num_queries_with_ground_truth": float(n),
    }

    if n == 0:
        out["mrr"] = 0.0
        for k in k_values:
            out[f"recall@{k}"] = 0.0
            out[f"precision@{k}"] = 0.0
            out[f"hit_rate@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        return out

    out["mrr"] = sum(r.reciprocal_rank for r in scored) / n
    for k in k_values:
        out[f"recall@{k}"] = sum(r.metrics_by_k[k].recall for r in scored) / n
        out[f"precision@{k}"] = sum(r.metrics_by_k[k].precision for r in scored) / n
        out[f"hit_rate@{k}"] = sum(r.metrics_by_k[k].hit_rate for r in scored) / n
        out[f"ndcg@{k}"] = sum(r.metrics_by_k[k].ndcg for r in scored) / n

    return out
