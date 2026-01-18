from typing import Any, Dict, List

from app.services.indexing.pinecone import PineconeStore
from app.services.indexing.schemas import SparseVector


class SparseRetriever:
    """
    Responsible for lexical retrieval using sparse vectors.
    """

    def __init__(self, store: PineconeStore):
        self.store = store

    async def search(
        self,
        sparse_vector: SparseVector,
        top_k: int,
        metadata_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Pure sparse retrieval (alpha = 0.0).
        """
        return await self._run(sparse_vector, top_k, metadata_filter)

    async def _run(
        self,
        sparse_vector: SparseVector,
        top_k: int,
        metadata_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        import anyio

        return await anyio.to_thread.run_sync(
            lambda: self.store.query(
                dense=[],
                sparse=sparse_vector,
                top_k=top_k,
                alpha=0.0,
                metadata_filter=metadata_filter,
            )
        )
