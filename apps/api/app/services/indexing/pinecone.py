import os
from typing import Any, Dict, List, Optional, Sequence

from pinecone import Pinecone

from .schemas import PineconeConfig, SparseVector
from .utils import apply_alpha_to_query

class PineconeStore:
    """
    Pinecone wrapper for dual-index mode (enforced).
    
    Uses separate indexes for patent and claim levels:
    - Patent index: For patent-level embeddings
    - Claim index: For claim-level embeddings
    - Automatically routes queries to the appropriate index based on level parameter
    
    Methods:
    - upsert(vectors, level): Insert vectors into appropriate index (level required)
    - query(dense, sparse, filter, alpha, level): Query appropriate index (level required)
    """

    def __init__(self, cfg: PineconeConfig):
        if not cfg.api_key:
            raise ValueError("Missing PINECONE_API_KEY")
        
        self.cfg = cfg
        self._pc = Pinecone(api_key=cfg.api_key)
        
        if not cfg.patent_index_host:
            raise ValueError("Missing PINECONE_PATENT_INDEX_HOST for dual-index mode")
        if not cfg.claim_index_host:
            raise ValueError("Missing PINECONE_CLAIM_INDEX_HOST for dual-index mode")
        
        self._patent_index = self._pc.Index(host=cfg.patent_index_host)
        self._claim_index = self._pc.Index(host=cfg.claim_index_host)

    @classmethod
    def from_env(cls) -> "PineconeStore":
        """Create PineconeStore from environment variables (dual-index mode)."""
        import os
        
        api_key = os.getenv("PINECONE_API_KEY", "")
        patent_index_host = os.getenv("PINECONE_PATENT_INDEX_HOST", "")
        claim_index_host = os.getenv("PINECONE_CLAIM_INDEX_HOST", "")
        namespace = os.getenv("PINECONE_NAMESPACE", "default")
        
        return cls(
            PineconeConfig(
                api_key=api_key,
                patent_index_host=patent_index_host,
                claim_index_host=claim_index_host,
                namespace=namespace,
            )
        )
    
    def _get_index(self, level: str):
        """
        Get the appropriate index based on level.
        
        Args:
            level: "patent" or "claim" (required)
            
        Returns:
            The appropriate Pinecone index
        """
        if level == "patent":
            return self._patent_index
        elif level == "claim":
            return self._claim_index
        else:
            raise ValueError(
                f"'level' must be 'patent' or 'claim', got: {level}"
            )

    def upsert(
        self, 
        vectors: List[Dict[str, Any]], 
        batch_size: int = 100,
        level: str = None,
    ) -> None:
        """
        Upsert vectors to the appropriate index.
        
        Args:
            vectors: List of vector dictionaries to upsert
            batch_size: Batch size for upserting
            level: "patent" or "claim" (required)
        """
        if not vectors:
            return
        
        index = self._get_index(level)
        
        for i in range(0, len(vectors), batch_size):
            index.upsert(
                vectors=vectors[i : i + batch_size], 
                namespace=self.cfg.namespace
            )

    def query(
        self,
        dense: Sequence[float],
        *,
        sparse: Optional[SparseVector] = None,
        top_k: int = 20,
        alpha: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        level: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the appropriate index.
        
        Args:
            dense: Dense query vector
            sparse: Optional sparse query vector
            top_k: Number of results to return
            alpha: Weight for dense vs sparse (1.0 = pure dense, 0.0 = pure sparse)
            metadata_filter: Metadata filters
            include_metadata: Whether to include metadata in results
            level: "patent" or "claim" (required)
            
        Returns:
            List of matches with id, score, and metadata
        """
        if not dense:
            raise ValueError("dense query vector required")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        index = self._get_index(level)
        q_dense, q_sparse = apply_alpha_to_query(dense, sparse, alpha)

        params: Dict[str, Any] = {
            "namespace": self.cfg.namespace,
            "top_k": top_k,
            "vector": q_dense,
            "include_metadata": include_metadata,
        }
        if metadata_filter:
            params["filter"] = metadata_filter
        if q_sparse is not None:
            params["sparse_vector"] = q_sparse

        res = index.query(**params)

        out: List[Dict[str, Any]] = []
        for m in (res.matches or []):
            out.append({"id": m.id, "score": m.score, "metadata": m.metadata or {}})
        return out
