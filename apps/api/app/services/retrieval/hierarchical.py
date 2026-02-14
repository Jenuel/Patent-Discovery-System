from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.sparse import SparseRetriever
from app.services.retrieval.fusion import to_scored_matches, fuse_rrf, ScoredMatch


@dataclass(frozen=True)
class HierarchicalConfig:
    """Configuration for hierarchical retrieval."""
    patent_top_k: int = 30          # Top patents after fusion
    claim_top_k: int = 60           # Top claims after fusion
    rrf_k: int = 60                 # RRF constant
    dense_top_k: int = 50           # Dense retrieval top-k
    sparse_top_k: int = 50          # Sparse retrieval top-k (patent-level only)


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

        # -----------------------
        # Stage 1: PATENT level
        # -----------------------
        patent_filter = {"level": "patent", **base_filter}

        # Dense patent retrieval (Pinecone)
        dense_pat = await self.dense.search(
            dense_vector=dense_query_vec,
            top_k=self.cfg.dense_top_k,
            metadata_filter=patent_filter,
        )

        # Sparse patent retrieval (Elasticsearch BM25)
        sparse_pat = []
        if self.sparse and query_text:
            sparse_pat = await self.sparse.search(
                query_text=query_text,
                top_k=self.cfg.sparse_top_k,
                metadata_filter=patent_filter,
            )

        # Fuse patent-level results using RRF
        fused_pat = fuse_rrf(
            to_scored_matches(dense_pat),
            to_scored_matches(sparse_pat),
            k=self.cfg.rrf_k,
            top_k=self.cfg.patent_top_k,
        )

        # Extract patent IDs from fused results
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
        # Only dense retrieval at claim level (no sparse)
        claim_filter = {
            "level": "claim",
            "patent_id": {"$in": patent_ids},
            **base_filter
        }

        dense_claim = await self.dense.search(
            dense_vector=dense_query_vec,
            top_k=self.cfg.claim_top_k,
            metadata_filter=claim_filter,
        )

        # Convert to ScoredMatch and return
        # No fusion needed at claim level since we only have dense results
        return to_scored_matches(dense_claim)
