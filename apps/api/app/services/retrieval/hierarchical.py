from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.sparse import SparseRetriever
from app.services.retrieval.fusion import to_scored_matches, ScoredMatch

log = get_logger(__name__)


@dataclass(frozen=True)
class HierarchicalConfig:
    """Configuration for hierarchical retrieval.

    Optimized for dataset sizes:
    - 113 patent-level instances (Qdrant patents_hybrid)
    - 2,200 claim-level instances (Qdrant claims_hybrid)
    """
    patent_top_k: int = 10
    claim_top_k: int = 30
    rrf_k: int = 30
    dense_top_k: int = 20
    sparse_top_k: int = 20


class HierarchicalRetriever:
    """
    2-step hierarchical retrieval backed entirely by Qdrant.

    Stage 1 — Patent level (``patents_hybrid``):
      - Qdrant-native hybrid search: dense + BM25 sparse, fused with RRF
        server-side inside Qdrant (via :meth:`DenseRetriever.search`).
      - No Python-level RRF needed — Qdrant returns a single ranked list.
      - Extracts patent_ids from top results.

    Stage 2 — Claim level (``claims_hybrid``):
      - Dense cosine search filtered by the patent_ids from Stage 1 *only*.
      - Returns top claim-level matches.

    Filter placement: ``base_filter`` is applied at Stage 1 and nowhere else.
    It describes patents, and Stage 2 only ever sees claims whose patent
    already passed it — so the selection is enforced exactly once, at the
    level whose payload schema actually has those fields.

    ``sparse`` is accepted for backward compatibility but is effectively
    unused when ``query_text`` is supplied to Stage 1 (``DenseRetriever``
    handles both arms natively).
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: Optional[SparseRetriever] = None,
        cfg: Optional[HierarchicalConfig] = None,
    ):
        self.dense = dense
        self.sparse = sparse
        self.cfg = cfg or HierarchicalConfig()

    async def retrieve_claims_hierarchical(
        self,
        *,
        dense_query_vec: List[float],
        query_text: Optional[str] = None,
        base_filter: Dict[str, Any],
    ) -> List[ScoredMatch]:
        """
        2-step hierarchical retrieval.

        Stage 1 (Patent-level):
          - Calls :meth:`DenseRetriever.search` with ``level="patent"`` and
            ``query_text``, which triggers Qdrant native hybrid search
            (dense + BM25 fused with RRF server-side).
          - Extracts top patent IDs.

        Stage 2 (Claim-level):
          - Calls :meth:`DenseRetriever.search` with ``level="claim"`` and a
            ``patent_id $in [...]`` filter built from Stage 1 results.
          - Returns top claim-level matches.

        Args:
            dense_query_vec: Dense embedding vector for semantic search.
            query_text:      Query text forwarded to BM25 arm (optional but
                             strongly recommended for hybrid quality).
            base_filter:     Patent-level metadata filters. Applied at Stage 1
                             only — see the class docstring.

        Returns:
            List of top claim-level :class:`~fusion.ScoredMatch` objects.
        """
        log.info("[HIERARCHICAL] Starting 2-stage hierarchical retrieval")
        log.debug(
            f"[HIERARCHICAL] Config: patent_top_k={self.cfg.patent_top_k}, "
            f"claim_top_k={self.cfg.claim_top_k}"
        )

        # ---------------------------------------------------------------
        # Stage 1: PATENT level — Qdrant hybrid (dense + BM25, native RRF)
        # ---------------------------------------------------------------
        log.info("[HIERARCHICAL STAGE 1] Starting patent-level hybrid retrieval")

        try:
            patent_results_raw = await self.dense.search(
                dense_vector=dense_query_vec,
                top_k=self.cfg.dense_top_k,
                metadata_filter=base_filter,
                level="patent",
                query_text=query_text,
            )
        except Exception:
            log.warning(
                "[HIERARCHICAL STAGE 1] Patent retrieval failed, returning empty results",
                exc_info=True,
            )
            return []

        patent_results = to_scored_matches(patent_results_raw)
        log.info(
            f"[HIERARCHICAL STAGE 1] Hybrid retrieval returned {len(patent_results)} patents"
        )

        # Extract patent IDs from the already-fused results (top patent_top_k)
        patent_ids = [
            m.metadata.get("patent_id")
            for m in patent_results[: self.cfg.patent_top_k]
            if m.metadata.get("patent_id")
        ]
        log.debug(
            f"[HIERARCHICAL STAGE 1] Extracted {len(patent_ids)} patent IDs: {patent_ids}"
        )

        if not patent_ids:
            log.warning("[HIERARCHICAL STAGE 1] No patent IDs found, returning empty results")
            return []

        # ---------------------------------------------------------------
        # Stage 2: CLAIM level — Qdrant dense, filtered by patent_ids
        # ---------------------------------------------------------------
        log.info("[HIERARCHICAL STAGE 2] Starting claim-level dense retrieval")

        claim_filter = {"patent_id": {"$in": patent_ids}}
        log.debug(f"[HIERARCHICAL STAGE 2] Claim filter: {claim_filter}")

        try:
            claim_results_raw = await self.dense.search(
                dense_vector=dense_query_vec,
                top_k=self.cfg.claim_top_k,
                metadata_filter=claim_filter,
                level="claim",
            )
        except Exception:
            log.warning(
                "[HIERARCHICAL STAGE 2] Claim retrieval failed, returning empty results",
                exc_info=True,
            )
            return []

        result = to_scored_matches(claim_results_raw)
        log.info(
            f"[HIERARCHICAL] Retrieval complete, returning {len(result)} claim-level matches"
        )
        return result