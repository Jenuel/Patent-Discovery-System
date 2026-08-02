"""Live retrieval evaluation runner.

Scores ``HierarchicalRetriever.retrieve_claims_hierarchical()`` (and, for
patent-selection metrics, its Stage-1 patent search directly) against a
ground-truth ``QueryCase`` set, using the pure IR metrics in
:mod:`evaluation.metrics`.

Bypasses ``MongoDBStore`` and ``GeminiClient`` entirely — retrieval
evaluation needs neither chunk text nor a generated answer.

Parameterisation
----------------
Every experiment in ``EVAL_ABLATION.md`` and ``EVAL_RERANKING.md`` was
originally run with throwaway scripts that are no longer in the repository, so
none of the documented tables could be reproduced. Those experiments are flag
combinations here instead:

===========================  =====================================
``--arm {hybrid,dense,bm25}``  Test 1 — is the BM25 arm earning its keep
``--fusion {rrf,dbsf}``        Test 3 — server-side fusion strategy
``--sparse-prefetch N``        Test 3 — sparse arm swept, dense held
``--dense-weight/--sparse-weight``  Test 4 — Python-level weighted RRF
``--depth N``                  Test 5 — recall ceiling past ``patent_top_k``
``--rerank / --rerank-model``  EVAL_RERANKING.md — cross-encoder sweep
``--tag NAME``                 keeps each run's CSVs instead of overwriting
===========================  =====================================

Except for the weighted-RRF diagnostic (see :func:`weighted_rrf`), every arm
runs through the ordinary production code path — the knobs live on
``DenseRetriever``/``HierarchicalConfig``, not on a parallel implementation.

Requires a live Qdrant instance with the corpus indexed.

Usage (run from ``apps/api``)::

    # production configuration
    python -m evaluation.retrieval_eval \\
        --queries evaluation/fixtures/queries.jsonl \\
        --output-dir evaluation/results

    # EVAL_ABLATION.md Test 1
    for arm in hybrid dense bm25; do
      python -m evaluation.retrieval_eval --queries evaluation/fixtures/queries.jsonl \\
        --output-dir evaluation/results --arm $arm --tag $arm
    done

    # EVAL_ABLATION.md Test 5 — recall ceiling
    python -m evaluation.retrieval_eval --queries evaluation/fixtures/queries.jsonl \\
        --output-dir evaluation/results --arm dense --depth 200 \\
        --patent-k 10,20,50,100,200 --tag ceiling
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import anyio
from dotenv import load_dotenv

from app.core.logging import get_logger
from app.services.indexing.embed import OpenAIEmbedder
from app.services.indexing.qdrant import QdrantHybridStore
from app.services.retrieval.dense import PATENT_ARMS, DenseRetriever
from app.services.retrieval.fusion import ScoredMatch, to_scored_matches
from app.services.retrieval.hierarchical import HierarchicalConfig, HierarchicalRetriever

from evaluation.dataset import QueryCase, load_query_cases
from evaluation.metrics import QueryResult, aggregate, evaluate_query

log = get_logger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    """One evaluation run's retrieval configuration.

    Defaults reproduce the production pipeline, so a no-flag run measures what
    users actually get. ``rerank`` defaults to ``None`` meaning "read
    ``settings.rerank_enabled``" — the harness should agree with production
    unless it is explicitly told otherwise.
    """

    arm: str = "hybrid"
    fusion: str = "rrf"
    prefetch_multiplier: int = 3
    dense_prefetch: Optional[int] = None
    sparse_prefetch: Optional[int] = None
    dense_weight: Optional[float] = None
    sparse_weight: Optional[float] = None
    rrf_k: int = 60
    patent_top_k: int = HierarchicalConfig.patent_top_k
    claim_top_k: int = HierarchicalConfig.claim_top_k
    depth: int = HierarchicalConfig.patent_top_k
    rerank: Optional[bool] = None
    rerank_model: str = ""
    tag: str = ""

    @property
    def weighted(self) -> bool:
        """True when Python-level weighted fusion replaces Qdrant's."""
        return self.dense_weight is not None or self.sparse_weight is not None


# ----------------------------------------------------------------------
# Pure helpers (unit-tested without Qdrant, OpenAI, or a model)
# ----------------------------------------------------------------------

def weighted_rrf(
    arms: Sequence[Tuple[float, Sequence[ScoredMatch]]],
    k: int = 60,
    limit: Optional[int] = None,
) -> List[ScoredMatch]:
    """Python-level weighted Reciprocal Rank Fusion over independent arms.

    ``score(d) = Σ_arm weight_arm / (k + rank_arm(d))``, rank starting at 1.

    This is the one path in the harness that does **not** run through
    production code, because Qdrant's native ``FusionQuery`` has no weight
    parameter — the only way to vary arm weights is to fetch each arm
    separately and fuse here. EVAL_ABLATION.md Test 4 exists precisely to show
    that no weighting beats dense-only, so this is a diagnostic, not a
    candidate configuration; results from it are not directly comparable to a
    native-fusion run (Qdrant applies the payload filter inside each
    sub-query, this applies it per arm).

    Ties resolve to first-seen order, so a deterministic input gives a
    deterministic ranking rather than one that depends on dict iteration.
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


def resolve_depth(cfg: EvalConfig, patent_k_values: Sequence[int]) -> int:
    """Patents to fetch and score at the patent level.

    Guards the defect where ``dedupe_patent_ids`` capped at ``patent_top_k``
    regardless of the requested k values, so ``--patent-k 20`` silently
    reported ``recall@20 == recall@10`` instead of retrieving 20 candidates.
    Any requested k deeper than the configured depth raises the depth rather
    than quietly truncating the measurement.
    """
    return max([cfg.depth, cfg.patent_top_k, *patent_k_values])


def report_paths(output_dir: Union[str, Path], tag: str = "") -> Tuple[Path, Path]:
    """Per-query and summary CSV paths for a run.

    Without ``--tag`` the historical filenames are kept. With one, each run
    writes its own pair — a sweep that overwrites a single
    ``retrieval_eval_summary.csv`` leaves nothing to compare against, which is
    how every table in EVAL_ABLATION.md ended up unreproducible.
    """
    if any(sep in tag for sep in ("/", "\\", "..")):
        raise ValueError(f"--tag must be a bare filename fragment, got {tag!r}")
    suffix = f"__{tag}" if tag else ""
    output_dir = Path(output_dir)
    return (
        output_dir / f"retrieval_eval_per_query{suffix}.csv",
        output_dir / f"retrieval_eval_summary{suffix}.csv",
    )


def config_columns(cfg: EvalConfig, reranked: bool, depth: int) -> Dict[str, Any]:
    """Configuration echoed into every summary row.

    Without these a CSV cannot say which arm, fusion, or reranker produced it,
    so two result files are indistinguishable once they leave the shell that
    ran them.
    """
    return {
        "arm": cfg.arm,
        "fusion": "weighted_rrf" if cfg.weighted else cfg.fusion,
        "reranked": reranked,
        "rerank_model": cfg.rerank_model if reranked else "",
        "patent_top_k": cfg.patent_top_k,
        "claim_top_k": cfg.claim_top_k,
        "depth": depth,
        "prefetch_multiplier": cfg.prefetch_multiplier,
        "dense_prefetch": cfg.dense_prefetch if cfg.dense_prefetch is not None else "",
        "sparse_prefetch": cfg.sparse_prefetch if cfg.sparse_prefetch is not None else "",
        "dense_weight": cfg.dense_weight if cfg.dense_weight is not None else "",
        "sparse_weight": cfg.sparse_weight if cfg.sparse_weight is not None else "",
        "rrf_k": cfg.rrf_k if cfg.weighted else "",
    }


# ----------------------------------------------------------------------
# Live pipeline
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Pipeline:
    embedder: OpenAIEmbedder
    qdrant: QdrantHybridStore
    retriever: HierarchicalRetriever
    cfg: EvalConfig
    depth: int

    @property
    def reranker(self) -> Optional[Any]:
        return self.retriever.reranker

    @property
    def fetch_depth(self) -> int:
        """Candidates pulled from Stage 1 before reranking.

        Mirrors ``retrieve_claims_hierarchical``'s ``stage1_top_k`` exactly:
        ``rerank_candidates`` with a reranker, ``dense_top_k`` without. Both
        branches matter — the hybrid arm's per-arm prefetch is a *multiple* of
        this number, so fetching a different depth than production hands
        Qdrant's fusion a different candidate pool and quietly measures a
        configuration nobody ships. The no-reranker branch previously returned
        ``depth``, which defaults to ``patent_top_k`` (10) rather than
        ``dense_top_k`` (20), halving the prefetch and moving patent MRR by
        more than any real effect this harness is meant to detect.

        Never fetches fewer than ``depth``, or the deepest requested k could
        not be scored.
        """
        if self.reranker:
            return max(self.depth, self.retriever.cfg.rerank_candidates)
        return max(self.depth, self.retriever.cfg.dense_top_k)


def build_reranker(cfg: EvalConfig) -> Optional[Any]:
    """Construct the reranker this run should use, or None.

    Mirrors ``RAGOrchestrator.__init__``: when ``--rerank/--no-rerank`` is not
    given, ``settings.rerank_enabled`` decides. The harness previously never
    built one at all, so it could not evaluate the production configuration
    with reranking on — awkward, given reranking ships disabled on the
    strength of an eval.
    """
    enabled = cfg.rerank
    if enabled is None:
        from app.core.settings import get_settings

        enabled = get_settings().rerank_enabled

    if not enabled:
        return None

    from app.services.rerank.reranker import CrossEncoderReranker, RerankConfig

    if cfg.rerank_model:
        return CrossEncoderReranker(RerankConfig(model_name=cfg.rerank_model))
    return CrossEncoderReranker.from_env()


async def build_pipeline(cfg: Optional[EvalConfig] = None, depth: Optional[int] = None) -> Pipeline:
    cfg = cfg or EvalConfig()
    embedder = OpenAIEmbedder.from_env()
    qdrant = QdrantHybridStore.from_env()
    reranker = build_reranker(cfg)

    dense = DenseRetriever(
        qdrant,
        arm=cfg.arm,
        fusion=cfg.fusion,
        prefetch_multiplier=cfg.prefetch_multiplier,
        dense_prefetch_limit=cfg.dense_prefetch,
        sparse_prefetch_limit=cfg.sparse_prefetch,
    )
    retriever = HierarchicalRetriever(
        dense=dense,
        cfg=HierarchicalConfig(
            patent_top_k=cfg.patent_top_k,
            claim_top_k=cfg.claim_top_k,
            rrf_k=cfg.rrf_k,
        ),
        reranker=reranker,
    )
    return Pipeline(
        embedder=embedder,
        qdrant=qdrant,
        retriever=retriever,
        cfg=cfg,
        depth=depth if depth is not None else cfg.depth,
    )


async def search_patents(
    pipeline: Pipeline, case: QueryCase, dense_query_vec: List[float]
) -> List[ScoredMatch]:
    """Stage 1 for patent-level scoring, on the configured arm."""
    cfg = pipeline.cfg
    fetch = pipeline.fetch_depth

    if cfg.weighted:
        dense_raw = await pipeline.qdrant.search_patents_dense(
            dense_query_vec, top_k=fetch, metadata_filter=case.metadata_filter or None
        )
        sparse_raw = await pipeline.qdrant.search_bm25(
            case.query, top_k=fetch, metadata_filter=case.metadata_filter or None
        )
        return weighted_rrf(
            [
                (cfg.dense_weight or 0.0, to_scored_matches(dense_raw)),
                (cfg.sparse_weight or 0.0, to_scored_matches(sparse_raw)),
            ],
            k=cfg.rrf_k,
            limit=fetch,
        )

    raw = await pipeline.retriever.dense.search(
        dense_vector=dense_query_vec,
        top_k=fetch,
        metadata_filter=case.metadata_filter,
        level="patent",
        query_text=case.query,
    )
    return to_scored_matches(raw)


async def search_claims(
    pipeline: Pipeline, case: QueryCase, dense_query_vec: List[float], patent_ids: List[str]
) -> List[ScoredMatch]:
    """Stage 2 for the weighted-fusion path only.

    Every other configuration goes through ``retrieve_claims_hierarchical``
    unmodified. Weighted fusion cannot: Qdrant has no weighted ``FusionQuery``,
    so Stage 1 had to happen out here, and reusing the production call would
    silently re-run an *unweighted* Stage 1 — reporting claim metrics for a
    configuration that was never requested.
    """
    if not patent_ids:
        return []
    raw = await pipeline.retriever.dense.search(
        dense_vector=dense_query_vec,
        top_k=pipeline.cfg.claim_top_k,
        metadata_filter={"patent_id": {"$in": patent_ids[: pipeline.cfg.patent_top_k]}},
        level="claim",
    )
    return to_scored_matches(raw)


async def run_query(pipeline: Pipeline, case: QueryCase) -> Tuple[List[str], List[str]]:
    """
    Returns (predicted_patent_ids, retrieved_claim_chunk_ids) for one query.

    Makes TWO retrieval calls:

      1. A patent-level Stage 1 on the configured arm, called independently so
         patent selection can be scored on its own terms. Inferring it only
         from which patent_ids survive into the final claim-level results would
         undercount: a correctly-selected patent can contribute zero claims to
         the top-K if another selected patent's claims outscore it.
      2. ``pipeline.retriever.retrieve_claims_hierarchical(...)`` — the real,
         unmodified production call — for claim-level metrics. (Weighted
         fusion is the one exception; see :func:`search_claims`.)

    **Stage 1b runs on both.** The independent patent path applies the same
    cross-encoder rerank ``retrieve_claims_hierarchical`` does, because
    otherwise a ``--rerank`` run reports pre-rerank ordering as patent metrics
    and post-rerank ordering as claim metrics, in one CSV, with nothing saying
    the two rows describe different pipelines.

    Both calls are wrapped in their own try/except: the direct Stage 1 call
    bypasses ``retrieve_claims_hierarchical``'s built-in fault tolerance (which
    catches exceptions and returns [] per stage), so ``run_query`` must not let
    one query's transient failure abort the whole batch. On exception, logs a
    warning and returns empty lists for that query rather than raising.
    """
    dense_query_vec = await anyio.to_thread.run_sync(lambda: pipeline.embedder.embed(case.query))

    try:
        patent_matches = await search_patents(pipeline, case, dense_query_vec)

        # Stage 1b — mirrors HierarchicalRetriever, including its "never fatal"
        # contract: a rerank failure degrades to retrieval order.
        if pipeline.reranker and patent_matches and case.query:
            try:
                patent_matches = await pipeline.reranker.rerank(
                    case.query, patent_matches, top_n=pipeline.depth
                )
            except Exception:
                log.warning(
                    f"[EVAL] Rerank failed for query_id={case.query_id!r}, "
                    f"continuing with retrieval order",
                    exc_info=True,
                )

        predicted_patent_ids = dedupe_patent_ids(patent_matches, limit=pipeline.depth)
    except Exception:
        log.warning(
            f"[EVAL] Patent-level retrieval failed for query_id={case.query_id!r}",
            exc_info=True,
        )
        predicted_patent_ids = []

    try:
        if pipeline.cfg.weighted:
            claim_matches = await search_claims(
                pipeline, case, dense_query_vec, predicted_patent_ids
            )
        else:
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
    config: Optional[Dict[str, Any]] = None,
    tag: str = "",
) -> Tuple[Path, Path]:
    """Writes the per-query and summary CSVs; returns their paths."""
    per_query_path, summary_path = report_paths(output_dir, tag)
    per_query_path.parent.mkdir(parents=True, exist_ok=True)

    per_query_rows = _per_query_rows(
        patent_results, "patent", patent_k_values, queries_by_id
    ) + _per_query_rows(claim_results, "claim", claim_k_values, queries_by_id)

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
        summary_rows.append({"level": level, **summary, **(config or {})})

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=union_fieldnames(summary_rows), restval=""
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    log.info(f"[EVAL] Wrote {per_query_path} and {summary_path}")
    return per_query_path, summary_path


def _parse_k_values(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", type=str, required=True, help="Path to queries.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for report CSVs")
    parser.add_argument("--patent-k", type=str, default="3,5,10", help="Comma-separated k values for patent-level metrics")
    parser.add_argument("--claim-k", type=str, default="5,10,20", help="Comma-separated k values for claim-level metrics")

    arm = parser.add_argument_group("retrieval arm (EVAL_ABLATION.md Tests 1, 3, 4)")
    arm.add_argument("--arm", choices=list(PATENT_ARMS), default="hybrid", help="Patent-level retrieval arm")
    arm.add_argument("--fusion", choices=["rrf", "dbsf"], default="rrf", help="Server-side fusion for the hybrid arm")
    arm.add_argument("--prefetch", type=int, default=3, help="Per-arm prefetch as a multiple of top_k")
    arm.add_argument("--dense-prefetch", type=int, default=None, help="Absolute dense-arm prefetch limit (overrides --prefetch)")
    arm.add_argument("--sparse-prefetch", type=int, default=None, help="Absolute BM25-arm prefetch limit (overrides --prefetch)")
    arm.add_argument("--dense-weight", type=float, default=None, help="Enable Python-level weighted RRF with this dense weight")
    arm.add_argument("--sparse-weight", type=float, default=None, help="Sparse weight for Python-level weighted RRF")
    arm.add_argument(
        "--rrf-k", type=int, default=60,
        help="RRF smoothing constant. Applies to --dense-weight/--sparse-weight "
             "fusion only; Qdrant's native RRF k is fixed server-side",
    )

    sizing = parser.add_argument_group("sizing")
    sizing.add_argument("--patent-top-k", type=int, default=HierarchicalConfig.patent_top_k, help="Patents passed from Stage 1 to Stage 2")
    sizing.add_argument("--claim-top-k", type=int, default=HierarchicalConfig.claim_top_k, help="Claims returned by Stage 2")
    sizing.add_argument(
        "--depth", type=int, default=None,
        help="Patents scored at the patent level (EVAL_ABLATION.md Test 5's "
             "recall ceiling). Defaults to the deepest --patent-k. Stage 1 "
             "always fetches at least what production would, so a shallow "
             "depth narrows what is scored, never what is retrieved",
    )

    rerank = parser.add_argument_group("reranking (EVAL_RERANKING.md)")
    rerank.add_argument("--rerank", dest="rerank", action="store_true", default=None, help="Force the cross-encoder on")
    rerank.add_argument("--no-rerank", dest="rerank", action="store_false", help="Force it off (default: settings.rerank_enabled)")
    rerank.add_argument("--rerank-model", type=str, default="", help="Cross-encoder model name")

    parser.add_argument("--tag", type=str, default="", help="Suffix the CSVs with __<tag> so a sweep keeps every run")
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        arm=args.arm,
        fusion=args.fusion,
        prefetch_multiplier=args.prefetch,
        dense_prefetch=args.dense_prefetch,
        sparse_prefetch=args.sparse_prefetch,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        rrf_k=args.rrf_k,
        patent_top_k=args.patent_top_k,
        claim_top_k=args.claim_top_k,
        depth=args.depth if args.depth is not None else HierarchicalConfig.patent_top_k,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        tag=args.tag,
    )


async def main() -> None:
    # ``get_settings()`` is what normally calls this, but ``build_pipeline``
    # constructs the embedder and Qdrant client *before* anything reads
    # settings — so without this a bare `python -m evaluation.retrieval_eval`
    # dies on "Missing OPENAI_API_KEY" despite a populated .env. Loaded here
    # rather than at import so that merely importing this module for its pure
    # helpers (see tests) stays free of environment side effects.
    load_dotenv()

    args = build_arg_parser().parse_args()

    patent_k_values = _parse_k_values(args.patent_k)
    claim_k_values = _parse_k_values(args.claim_k)
    cfg = config_from_args(args)

    if cfg.weighted and cfg.arm != "hybrid":
        raise SystemExit("--dense-weight/--sparse-weight fuse two arms; use --arm hybrid")

    depth = resolve_depth(cfg, patent_k_values)
    report_paths(args.output_dir, cfg.tag)  # fail fast on a bad --tag

    cases = load_query_cases(args.queries)
    log.info(f"[EVAL] Loaded {len(cases)} query cases from {args.queries}")

    pipeline = await build_pipeline(cfg, depth=depth)
    log.info(
        f"[EVAL] arm={cfg.arm} fusion={'weighted_rrf' if cfg.weighted else cfg.fusion} "
        f"rerank={bool(pipeline.reranker)} depth={depth} "
        f"fetch_depth={pipeline.fetch_depth} "
        f"patent_top_k={cfg.patent_top_k} claim_top_k={cfg.claim_top_k}"
    )
    try:
        patent_results, claim_results = await run_evaluation(
            pipeline, cases, patent_k_values=patent_k_values, claim_k_values=claim_k_values
        )
        queries_by_id = {c.query_id: c.query for c in cases}
        write_reports(
            args.output_dir,
            patent_results,
            claim_results,
            patent_k_values,
            claim_k_values,
            queries_by_id,
            config=config_columns(cfg, reranked=bool(pipeline.reranker), depth=depth),
            tag=cfg.tag,
        )
    finally:
        await pipeline.qdrant.close()

    log.info("[EVAL] Evaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
