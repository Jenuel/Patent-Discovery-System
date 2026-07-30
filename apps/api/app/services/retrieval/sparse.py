from typing import Any, Dict, List

from app.services.indexing.qdrant import QdrantHybridStore


class SparseRetriever:
    """
    BM25-only retrieval backed by Qdrant ``patents_hybrid``.

    Operates at PATENT LEVEL ONLY.

    .. note::
        With :class:`~app.services.retrieval.dense.DenseRetriever` now calling
        :meth:`~QdrantHybridStore.search_hybrid` for patent-level retrieval,
        this retriever is only needed as a BM25-only fallback (e.g. when no
        dense embedding is available).  Normal retrieval paths go through
        ``DenseRetriever`` directly.
    """

    def __init__(self, store: QdrantHybridStore):
        self.store = store

    async def search(
        self,
        query_text: str,
        top_k: int,
        metadata_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Pure BM25 sparse retrieval at patent level.

        Args:
            query_text:      Query text for BM25 search.
            top_k:           Number of results to return.
            metadata_filter: Metadata filters.

        Returns:
            List of patent-level search results.
        """
        return await self.store.search_bm25(
            query_text=query_text,
            top_k=top_k,
            metadata_filter=metadata_filter,
            search_fields=["title^2", "abstract", "text", "claims", "patent_id"],
        )
