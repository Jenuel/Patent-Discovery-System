from typing import Any, Dict, List

from app.services.indexing.pinecone import PineconeStore
from app.services.indexing.schemas import SparseVector


class DenseRetriever:
    """
    Responsible for semantic-only retrieval.
    """

    def __init__(self, store: PineconeStore):
        self.store = store

    async def search(
        self,
        dense_vector: List[float],
        top_k: int,
        metadata_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Pure dense retrieval (alpha = 1.0, no sparse).
        """
        return await self._run(dense_vector, top_k, metadata_filter)

    async def _run(
        self,
        dense_vector: List[float],
        top_k: int,
        metadata_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        import anyio
        # Extract level for index routing (used in dual-index mode)
        level = metadata_filter.get("level")

        return await anyio.to_thread.run_sync(
            lambda: self.store.query(
                dense=dense_vector,
                sparse=None,
                top_k=top_k,
                alpha=1.0,
                metadata_filter=metadata_filter,
                level=level,
            )
        )
