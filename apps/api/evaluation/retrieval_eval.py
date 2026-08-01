"""Live retrieval evaluation runner.

Scores ``HierarchicalRetriever.retrieve_claims_hierarchical()`` (and, for
patent-selection metrics, its Stage-1 patent search directly) against a
ground-truth ``QueryCase`` set, using the pure IR metrics in
:mod:`evaluation.metrics`.

Bypasses ``MongoDBStore`` and ``GeminiClient`` entirely — retrieval
evaluation needs neither chunk text nor a generated answer.

Requires a live Qdrant instance with the example corpus indexed. See
``RETRIEVAL_EVAL_PLAN.md`` / the module docstring commands below for the
manual indexing steps.

Usage (run from ``apps/api``)::

    python -m evaluation.retrieval_eval \\
        --queries evaluation/fixtures/queries.jsonl \\
        --output-dir evaluation/results \\
        --patent-k 3,5,10 --claim-k 5,10,20
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import anyio

from app.core.logging import get_logger
from app.services.indexing.embed import OpenAIEmbedder
from app.services.indexing.qdrant import QdrantHybridStore
from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.fusion import ScoredMatch, to_scored_matches
from app.services.retrieval.hierarchical import HierarchicalConfig, HierarchicalRetriever

from evaluation.dataset import QueryCase, load_query_cases
from evaluation.metrics import QueryResult, aggregate, evaluate_query

log = get_logger(__name__)


@dataclass(frozen=True)
class Pipeline:
    embedder: OpenAIEmbedder
    qdrant: QdrantHybridStore
    retriever: HierarchicalRetriever


async def build_pipeline() -> Pipeline:
    embedder = OpenAIEmbedder.from_env()
    qdrant = QdrantHybridStore.from_env()
    
    retriever = HierarchicalRetriever(
        dense=DenseRetriever(qdrant),
        cfg=HierarchicalConfig(),
    )
    return Pipeline(embedder=embedder, qdrant=qdrant, retriever=retriever)


def dedupe_patent_ids(matches: Sequence[ScoredMatch], limit: int) -> List[str]:
    """Order-preserving distinct patent_id extraction, capped at `limit`.

    Matches with no `patent_id` in metadata are skipped.
    """
    seen: set = set()
    out: List[str] = []
    for match in matches:
        patent_id = match.metadata.get("patent_id")
        if not patent_id or patent_id in seen:
            continue
        seen.add(patent_id)
        out.append(patent_id)
        if len(out) >= limit:
            break
    return out


async def run_query(pipeline: Pipeline, case: QueryCase) -> Tuple[List[str], List[str]]:
    """
    Returns (predicted_patent_ids, retrieved_claim_chunk_ids) for one query.

    Makes TWO retrieval calls:

      1. ``pipeline.retriever.dense.search(level="patent", ...)`` directly —
         duplicates (does not replace) Stage 1 of
         ``retrieve_claims_hierarchical``, called independently so patent
         selection can be scored on its own terms. Inferring it only from
         which patent_ids survive into the final claim-level results would
         undercount: a correctly-selected patent can contribute zero claims
         to the top-K if another selected patent's claims outscore it.
      2. ``pipeline.retriever.retrieve_claims_hierarchical(...)`` — the real,
         unmodified production call — for claim-level metrics.

    Both calls are wrapped in their own try/except: the direct dense.search
    call above bypasses ``retrieve_claims_hierarchical``'s built-in fault
    tolerance (which catches exceptions and returns [] per stage), so
    ``run_query`` must not let one query's transient failure abort the whole
    batch. On exception, logs a warning and returns empty lists for that
    query rather than raising.
    """
    dense_query_vec = await anyio.to_thread.run_sync(lambda: pipeline.embedder.embed(case.query))
    cfg = pipeline.retriever.cfg

    try:
        patent_results_raw = await pipeline.retriever.dense.search(
            dense_vector=dense_query_vec,
            top_k=cfg.dense_top_k,
            metadata_filter=case.metadata_filter,
            level="patent",
            query_text=case.query,
        )
        patent_matches = to_scored_matches(patent_results_raw)
        predicted_patent_ids = dedupe_patent_ids(patent_matches, limit=cfg.patent_top_k)
    except Exception:
        log.warning(
            f"[EVAL] Patent-level retrieval failed for query_id={case.query_id!r}",
            exc_info=True,
        )
        predicted_patent_ids = []

    try:
        claim_matches = await pipeline.retriever.retrieve_claims_hierarchical(
            dense_query_vec=dense_query_vec,
            query_text=case.query,
            base_filter=case.metadata_filter,
        )
        retrieved_claim_chunk_ids = [m.id for m in claim_matches]
    except Exception:
        log.warning(
            f"[EVAL] Claim-level retrieval failed for query_id={case.query_id!r}",
            exc_info=True,
        )
        retrieved_claim_chunk_ids = []

    return predicted_patent_ids, retrieved_claim_chunk_ids


async def run_evaluation(
    pipeline: Pipeline,
    cases: List[QueryCase],
    patent_k_values: Sequence[int] = (3, 5, 10),
    claim_k_values: Sequence[int] = (5, 10, 20),
) -> Tuple[List[QueryResult], List[QueryResult]]:
    """Returns (patent_level_results, claim_level_results), one QueryResult
    per case per level. Per-query failures (see run_query) are recorded,
    not raised."""
    patent_results: List[QueryResult] = []
    claim_results: List[QueryResult] = []

    for case in cases:
        predicted_patent_ids, retrieved_claim_chunk_ids = await run_query(pipeline, case)

        patent_results.append(
            evaluate_query(
                predicted_patent_ids,
                case.relevant_patent_ids,
                k_values=patent_k_values,
                query_id=case.query_id,
            )
        )
        claim_results.append(
            evaluate_query(
                retrieved_claim_chunk_ids,
                case.relevant_chunk_id_set,
                k_values=claim_k_values,
                relevance_grades=case.relevant_chunk_ids,
                query_id=case.query_id,
            )
        )

    return patent_results, claim_results


def _per_query_rows(
    results: List[QueryResult], level: str, k_values: Sequence[int], queries_by_id: Dict[str, str]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in results:
        row: Dict[str, object] = {
            "query_id": r.query_id,
            "level": level,
            "query": queries_by_id.get(r.query_id, ""),
            "mrr": r.reciprocal_rank,
            "has_ground_truth": r.has_ground_truth,
            "num_relevant": r.num_relevant,
            "num_retrieved": r.num_retrieved,
        }
        for k in k_values:
            m = r.metrics_by_k[k]
            row[f"recall@{k}"] = m.recall
            row[f"precision@{k}"] = m.precision
            row[f"hit_rate@{k}"] = m.hit_rate
            row[f"ndcg@{k}"] = m.ndcg
        rows.append(row)
    return rows


def union_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """
    Column union across all rows, first-seen order preserved.

    Patent and claim rows carry *different* ``@k`` columns by design — patent
    k stops at 10 because ``HierarchicalConfig.patent_top_k`` is 10, while
    claim k runs to 20 under a ``claim_top_k`` of 30. So no single row's keys
    are a superset of the rest, and taking fieldnames from ``rows[0]`` makes
    ``csv.DictWriter`` raise on every column the first row happens to lack.
    Pair this with ``restval=""`` so a level that has no value for a given k
    writes an empty cell instead of failing.
    """
    seen: Dict[str, None] = {}
    for row in rows:
        for key in row:
            seen[key] = None
    return list(seen)


def write_reports(
    output_dir: Union[str, Path],
    patent_results: List[QueryResult],
    claim_results: List[QueryResult],
    patent_k_values: Sequence[int],
    claim_k_values: Sequence[int],
    queries_by_id: Dict[str, str],
) -> None:
    """Writes retrieval_eval_per_query.csv and retrieval_eval_summary.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_query_rows = _per_query_rows(
        patent_results, "patent", patent_k_values, queries_by_id
    ) + _per_query_rows(claim_results, "claim", claim_k_values, queries_by_id)

    per_query_path = output_dir / "retrieval_eval_per_query.csv"
    with open(per_query_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = union_fieldnames(per_query_rows) or [
            "query_id", "level", "query", "mrr", "has_ground_truth",
            "num_relevant", "num_retrieved",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(per_query_rows)

    summary_rows = []
    for level, results, k_values in (
        ("patent", patent_results, patent_k_values),
        ("claim", claim_results, claim_k_values),
    ):
        summary = aggregate(results, k_values)
        summary_rows.append({"level": level, **summary})

    summary_path = output_dir / "retrieval_eval_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=union_fieldnames(summary_rows), restval=""
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    log.info(f"[EVAL] Wrote {per_query_path} and {summary_path}")


def _parse_k_values(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against ground truth.")
    parser.add_argument("--queries", type=str, required=True, help="Path to queries.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for report CSVs")
    parser.add_argument("--patent-k", type=str, default="3,5,10", help="Comma-separated k values for patent-level metrics")
    parser.add_argument("--claim-k", type=str, default="5,10,20", help="Comma-separated k values for claim-level metrics")
    args = parser.parse_args()

    patent_k_values = _parse_k_values(args.patent_k)
    claim_k_values = _parse_k_values(args.claim_k)

    cases = load_query_cases(args.queries)
    log.info(f"[EVAL] Loaded {len(cases)} query cases from {args.queries}")

    pipeline = await build_pipeline()
    try:
        patent_results, claim_results = await run_evaluation(
            pipeline, cases, patent_k_values=patent_k_values, claim_k_values=claim_k_values
        )
        queries_by_id = {c.query_id: c.query for c in cases}
        write_reports(
            args.output_dir, patent_results, claim_results, patent_k_values, claim_k_values, queries_by_id
        )
    finally:
        await pipeline.qdrant.close()

    log.info("[EVAL] Evaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
