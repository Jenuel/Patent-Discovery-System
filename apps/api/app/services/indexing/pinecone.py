import os
from typing import Any, Dict, List, Optional, Sequence

from pinecone import Pinecone

from .schemas import PineconeConfig, SparseVector
from .utils import apply_alpha_to_query

class PineconeStore:
    """
    A thin Pinecone wrapper:
    - upsert(vectors)
    - query(dense, sparse, filter, alpha)
    """

    def __init__(self, cfg: PineconeConfig):
        if not cfg.api_key:
            raise ValueError("Missing PINECONE_API_KEY")
        if not cfg.index_host:
            raise ValueError("Missing PINECONE_INDEX_HOST")
        self.cfg = cfg
        self._pc = Pinecone(api_key=cfg.api_key)
        self._index = self._pc.Index(host=cfg.index_host)

    @classmethod
    def from_env(cls) -> "PineconeStore":
        return cls(
            PineconeConfig(
                api_key=os.getenv("PINECONE_API_KEY", ""),
                index_host=os.getenv("PINECONE_INDEX_HOST", ""),
                namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            )
        )

    def upsert(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> None:
        if not vectors:
            return
        for i in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[i : i + batch_size], namespace=self.cfg.namespace)

    def query(
        self,
        dense: Sequence[float],
        *,
        sparse: Optional[SparseVector] = None,
        top_k: int = 20,
        alpha: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        if not dense:
            raise ValueError("dense query vector required")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

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

        res = self._index.query(**params)

        out: List[Dict[str, Any]] = []
        for m in (res.matches or []):
            out.append({"id": m.id, "score": m.score, "metadata": m.metadata or {}})
        return out