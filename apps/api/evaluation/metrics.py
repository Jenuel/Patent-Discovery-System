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
  - ``recall_at_k`` divides by the *full* relevant-set size, which is the
    standard definition but caps the metric below 1.0 whenever ``|R| > k``.
    ``aggregate`` therefore reports ``recall_ceiling@k`` beside every
    ``recall@k`` so the pair can be read together — see
    :func:`recall_ceiling_at_k`.
  - ``aggregate`` emits a percentile-bootstrap confidence interval for every
    mean it reports. At the fixture set's n=19 the interval is not a footnote
    to the comparison, it *is* the comparison.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import AbstractSet, Dict, List, Mapping, Optional, Sequence, Tuple

# Bootstrap defaults. The seed is fixed so that re-running an evaluation
# reproduces its own intervals exactly; an interval that drifted between runs
# would be indistinguishable from the effect it exists to rule out.
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SEED = 12345


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def recall_ceiling_at_k(num_relevant: int, k: int) -> float:
    """The largest ``recall@k`` this query could possibly score.

    ``recall_at_k`` divides by ``len(relevant_ids)``, so a query with 8
    relevant patents cannot exceed ``3/8 = 0.375`` at k=3 however good the
    retriever is. Reporting ``recall@3 = 0.190`` against an implied maximum of
    1.0 understates the system by roughly a factor of three; reporting it
    beside a ceiling of 0.582 does not.

    ``precision_at_k`` already carries the analogous correction in its
    denominator. Recall cannot fix it the same way — dividing by ``min(k,
    |R|)`` would silently redefine the metric into R-precision at low k — so
    the ceiling is reported alongside instead of folded in.
    """
    if num_relevant <= 0:
        return 0.0
    return min(k, num_relevant) / num_relevant


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


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of an **already sorted** sequence.

    ``pct`` is a fraction in [0, 1]. Interpolates rather than picking the
    nearest sample so that a small ``num_samples`` does not quantise the
    interval into visible steps.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = pct * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def bootstrap_cis(
    columns: Mapping[str, Sequence[float]],
    num_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Tuple[float, float]]:
    """Percentile-bootstrap confidence interval for the mean of each column.

    ``columns`` maps a metric name to its per-query values, all of equal
    length — one entry per scored query, in a consistent query order.

    **Every column is resampled with one shared set of query-index draws.**
    Resampling each metric independently would produce intervals describing
    different hypothetical query sets, so ``recall@10``'s interval could not
    be read against ``mrr``'s. Sharing the draws keeps them mutually
    comparable, and costs one pass instead of one per metric.

    Deterministic given ``seed``. With a single scored query the interval
    collapses onto that query's value — bootstrap cannot manufacture spread
    from one observation, and a zero-width interval is the honest report.
    """
    names = list(columns)
    if not names:
        return {}

    n = len(columns[names[0]])
    for name in names:
        if len(columns[name]) != n:
            raise ValueError(
                f"all columns must have one value per query; {name!r} has "
                f"{len(columns[name])}, expected {n}"
            )
    if n == 0:
        return {name: (0.0, 0.0) for name in names}
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    sampled_means: Dict[str, List[float]] = {name: [] for name in names}

    for _ in range(num_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        for name in names:
            values = columns[name]
            sampled_means[name].append(sum(values[i] for i in idx) / n)

    lo_pct = (1.0 - confidence) / 2.0
    out: Dict[str, Tuple[float, float]] = {}
    for name in names:
        ordered = sorted(sampled_means[name])
        out[name] = (
            _percentile(ordered, lo_pct),
            _percentile(ordered, 1.0 - lo_pct),
        )
    return out


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


def per_query_columns(
    scored: Sequence[QueryResult], k_values: Sequence[int]
) -> Dict[str, List[float]]:
    """Per-query value vectors for every metric ``aggregate`` reports.

    One dict entry per metric, each a list with one value per scored query in
    the given order. Both the means and the bootstrap draw from this, so a
    reported mean and its interval can never describe different numbers.
    """
    columns: Dict[str, List[float]] = {"mrr": [r.reciprocal_rank for r in scored]}
    for k in k_values:
        columns[f"recall@{k}"] = [r.metrics_by_k[k].recall for r in scored]
        columns[f"recall_ceiling@{k}"] = [
            recall_ceiling_at_k(r.num_relevant, k) for r in scored
        ]
        columns[f"precision@{k}"] = [r.metrics_by_k[k].precision for r in scored]
        columns[f"hit_rate@{k}"] = [r.metrics_by_k[k].hit_rate for r in scored]
        columns[f"ndcg@{k}"] = [r.metrics_by_k[k].ndcg for r in scored]
    return columns


def aggregate(
    results: Sequence[QueryResult],
    k_values: Sequence[int],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Mean MRR + mean recall/precision/hit_rate/ndcg per k, averaged only
    over queries where has_ground_truth is True.

    Each mean is accompanied by:

      - ``recall_ceiling@k`` — the mean of the largest recall@k each query
        could attain (:func:`recall_ceiling_at_k`). ``recall@k`` alone reads as
        a fraction of 1.0, which it is not whenever ``|R| > k``.
      - ``<metric>_ci_lo`` / ``<metric>_ci_hi`` — a percentile-bootstrap
        interval on the mean. Every documented comparison in EVAL_ABLATION.md
        and EVAL_RERANKING.md re-derived "n=19, 95% CI ≈ ±0.15" by hand;
        emitting it here means a future sweep cannot quietly skip the check.

    Keys: "mrr", "recall@5", "recall_ceiling@5", "precision@5", "hit_rate@5",
    "ndcg@5", ... each plus "_ci_lo"/"_ci_hi", then "num_queries" and
    "num_queries_with_ground_truth".

    Pass ``bootstrap_samples=0`` to skip the resampling entirely; the interval
    columns are still emitted (as 0.0) so that a run with it disabled produces
    the same CSV schema as one without.
    """
    scored = [r for r in results if r.has_ground_truth]
    n = len(scored)

    out: Dict[str, float] = {
        "num_queries": float(len(results)),
        "num_queries_with_ground_truth": float(n),
    }

    columns = per_query_columns(scored, k_values)

    for name, values in columns.items():
        out[name] = sum(values) / n if n else 0.0

    if n and bootstrap_samples > 0:
        cis = bootstrap_cis(
            columns,
            num_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
    else:
        cis = {name: (0.0, 0.0) for name in columns}

    for name, (lo, hi) in cis.items():
        out[f"{name}_ci_lo"] = lo
        out[f"{name}_ci_hi"] = hi

    return out
