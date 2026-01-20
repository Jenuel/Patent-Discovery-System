from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.sparse import SparseRetriever
from app.services.retrieval.fusion import to_scored_matches, fuse_rrf, ScoredMatch
from app.services.indexing.pinecone import SparseVector


@dataclass(frozen=True)
class HierarchicalConfig:
    patent_top_k: int = 30          
    claim_top_k: int = 60           
    rrf_k: int = 60                 
    dense_top_k: int = 50           
    sparse_top_k: int = 50          


class HierarchicalRetriever:
    """
    Uses:
      - dense.py (semantic retrieval)
      - sparse.py (lexical retrieval)
      - fusion.py (combine ranked lists)
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
        sparse_query_vec: Optional[SparseVector],
        base_filter: Dict[str, Any],        
    ) -> List[ScoredMatch]:
        """
        Stage 1: patent-level retrieve -> fuse -> pick patent_ids
        Stage 2: claim-level retrieve within patent_ids -> fuse -> return claims
        """

        # -----------------------
        # Stage 1: PATENT level
        # -----------------------
        patent_filter = {"level": "patent", **base_filter}

        dense_pat = await self.dense.search(
            dense_vector=dense_query_vec,
            top_k=self.cfg.dense_top_k,
            metadata_filter=patent_filter,
        )

        sparse_pat = []
        if self.sparse and sparse_query_vec:
            sparse_pat = await self.sparse.search(
                sparse_vector=sparse_query_vec,
                top_k=self.cfg.sparse_top_k,
                metadata_filter=patent_filter,
            )

        fused_pat = fuse_rrf(
            to_scored_matches(dense_pat),
            to_scored_matches(sparse_pat),
            k=self.cfg.rrf_k,
            top_k=self.cfg.patent_top_k,
        )

        patent_ids = [
            m.metadata.get("patent_id")
            for m in fused_pat
            if m.metadata.get("patent_id")
        ]

        if not patent_ids:
            return []

        # -----------------------
        # Stage 2: CLAIM level
        # -----------------------
        claim_filter = {"level": "claim", "patent_id": {"$in": patent_ids}, **base_filter}

        dense_claim = await self.dense.search(
            dense_vector=dense_query_vec,
            top_k=self.cfg.dense_top_k,
            metadata_filter=claim_filter,
        )

        sparse_claim = []
        if self.sparse and sparse_query_vec:
            sparse_claim = await self.sparse.search(
                sparse_vector=sparse_query_vec,
                top_k=self.cfg.sparse_top_k,
                metadata_filter=claim_filter,
            )

        fused_claim = fuse_rrf(
            to_scored_matches(dense_claim),
            to_scored_matches(sparse_claim),
            k=self.cfg.rrf_k,
            top_k=self.cfg.claim_top_k,
        )

        return fused_claim
