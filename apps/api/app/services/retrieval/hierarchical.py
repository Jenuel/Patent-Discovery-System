from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.sparse import SparseRetriever
from app.services.retrieval.fusion import to_scored_matches, fuse_rrf, ScoredMatch

log = get_logger(__name__)


@dataclass(frozen=True)
class HierarchicalConfig:
    """Configuration for hierarchical retrieval.
    
    Optimized for dataset sizes:
    - 113 patent-level instances (Pinecone + Elasticsearch)
    - 2,200 claim-level instances (Pinecone + MongoDB)
    """
    patent_top_k: int = 10          
    claim_top_k: int = 30           
    rrf_k: int = 30                 
    dense_top_k: int = 20          
    sparse_top_k: int = 20          


class HierarchicalRetriever:
    """
    2-step hierarchical retrieval:
    
    Dense (Pinecone):
      - Stage 1: Retrieve top patents using dense vectors
      - Stage 2: Retrieve top claims from selected patents using dense vectors
    
    Sparse (Elasticsearch BM25):
      - Stage 1 only: Retrieve top patents using BM25 lexical search
      - No claim-level sparse retrieval
    
    Fusion:
      - Stage 1: Fuse dense + sparse patent results using RRF
      - Stage 2: Use only dense claim results (no sparse at claim level)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: Optional[SparseRetriever],
        cfg: Optional[HierarchicalConfig] = None,
    ):
        self.dense = dense
        self.sparse = sparse
        self.cfg = cfg or HierarchicalConfig()

    async def retrieve_claims_hierarchical(
        self,
        *,
        dense_query_vec: List[float],
        query_text: Optional[str] = None,  # For sparse BM25 search
        base_filter: Dict[str, Any],
    ) -> List[ScoredMatch]:
        """
        2-step hierarchical retrieval:
        
        Stage 1 (Patent-level):
          - Dense retrieval via Pinecone
          - Sparse retrieval via Elasticsearch BM25 (if query_text provided)
          - Fuse results using RRF
          - Extract patent IDs
        
        Stage 2 (Claim-level):
          - Dense retrieval only via Pinecone (filtered by patent IDs from Stage 1)
          - Return top claims
        
        Args:
            dense_query_vec: Dense embedding vector for semantic search
            query_text: Query text for BM25 sparse search (optional)
            base_filter: Base metadata filters
            
        Returns:
            List of top claim-level matches
        """
        log.info("[HIERARCHICAL] Starting 2-stage hierarchical retrieval")
        log.debug(f"[HIERARCHICAL] Config: patent_top_k={self.cfg.patent_top_k}, claim_top_k={self.cfg.claim_top_k}, rrf_k={self.cfg.rrf_k}")

        # -----------------------
        # Stage 1: PATENT level
        # -----------------------
        log.info("[HIERARCHICAL STAGE 1] Starting patent-level retrieval")
        patent_filter = {"level": "patent", **base_filter}
        log.debug(f"[HIERARCHICAL STAGE 1] Patent filter: {patent_filter}")

        # Dense patent retrieval (Pinecone)
        log.debug(f"[HIERARCHICAL STAGE 1] Performing dense retrieval (top_k={self.cfg.dense_top_k})")
        dense_pat = await self.dense.search(
            dense_vector=dense_query_vec,
            top_k=self.cfg.dense_top_k,
            metadata_filter=patent_filter,
        )
        log.info(f"[HIERARCHICAL STAGE 1] Dense retrieval found {len(dense_pat)} patents")

        # Sparse patent retrieval (Elasticsearch BM25)
        sparse_pat = []
        if self.sparse and query_text:
            log.debug(f"[HIERARCHICAL STAGE 1] Performing sparse BM25 retrieval (top_k={self.cfg.sparse_top_k})")
            sparse_pat = await self.sparse.search(
                query_text=query_text,
                top_k=self.cfg.sparse_top_k,
                metadata_filter=patent_filter,
            )
            log.info(f"[HIERARCHICAL STAGE 1] Sparse retrieval found {len(sparse_pat)} patents")
        else:
            log.debug("[HIERARCHICAL STAGE 1] Skipping sparse retrieval (no sparse retriever or query text)")

        # Fuse patent-level results using RRF
        log.debug(f"[HIERARCHICAL STAGE 1] Fusing dense and sparse results with RRF (k={self.cfg.rrf_k})")
        fused_pat = fuse_rrf(
            to_scored_matches(dense_pat),
            to_scored_matches(sparse_pat),
            k=self.cfg.rrf_k,
            top_k=self.cfg.patent_top_k,
        )
        log.info(f"[HIERARCHICAL STAGE 1] Fusion complete, selected top {len(fused_pat)} patents")

        # Extract patent IDs from fused results
        patent_ids = [
            m.metadata.get("patent_id")
            for m in fused_pat
            if m.metadata.get("patent_id")
        ]
        log.debug(f"[HIERARCHICAL STAGE 1] Extracted {len(patent_ids)} patent IDs: {patent_ids}")

        if not patent_ids:
            log.warning("[HIERARCHICAL STAGE 1] No patent IDs found, returning empty results")
            return []

        # -----------------------
        # Stage 2: CLAIM level
        # -----------------------
        log.info("[HIERARCHICAL STAGE 2] Starting claim-level retrieval")
        # Only dense retrieval at claim level (no sparse)
        claim_filter = {
            "level": "claim",
            "patent_id": {"$in": patent_ids},
            **base_filter
        }
        log.debug(f"[HIERARCHICAL STAGE 2] Claim filter: {claim_filter}")

        log.debug(f"[HIERARCHICAL STAGE 2] Performing dense retrieval (top_k={self.cfg.claim_top_k})")
        dense_claim = await self.dense.search(
            dense_vector=dense_query_vec,
            top_k=self.cfg.claim_top_k,
            metadata_filter=claim_filter,
        )
        log.info(f"[HIERARCHICAL STAGE 2] Dense retrieval found {len(dense_claim)} claims")

        # Convert to ScoredMatch and return
        # No fusion needed at claim level since we only have dense results
        result = to_scored_matches(dense_claim)
        log.info(f"[HIERARCHICAL] Hierarchical retrieval complete, returning {len(result)} claim-level matches")
        return result