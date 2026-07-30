from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.indexing.qdrant import QdrantHybridStore


class DenseRetriever:
    """
    Responsible for dense (and hybrid) retrieval backed entirely by Qdrant.

    Routing by ``level``:
    - ``"patent"`` → :meth:`~QdrantHybridStore.search_hybrid` on ``patents_hybrid``
      (Qdrant-native Prefetch + RRF fusion over dense + BM25 vectors).
    - ``"claim"``  → :meth:`~QdrantHybridStore.search_claims_dense` on ``claims_hybrid``
      (pure dense cosine search, optionally filtered by patent_id).

    Both methods return the unified ``[{id, score, metadata}]`` schema.
    """

    def __init__(self, store: QdrantHybridStore):
        self.store = store

    async def search(
        self,
        dense_vector: List[float],
        top_k: int,
        metadata_filter: Dict[str, Any],
        level: str,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Route to the appropriate Qdrant collection based on ``level``.

        Args:
            dense_vector:    Pre-computed OpenAI dense query embedding.
            top_k:           Number of results to return.
            metadata_filter: Payload filters (equality, range, or ``$in``).
            level:           ``"patent"`` or ``"claim"``.
            query_text:      Required for patent-level hybrid search (BM25 arm).
                             Ignored for claim-level dense search.

        Returns:
            List of dicts with ``id``, ``score``, and ``metadata``.
        """
        if level == "patent":
            return await self._search_patents(
                dense_vector=dense_vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                query_text=query_text,
            )
        elif level == "claim":
            return await self._search_claims(
                dense_vector=dense_vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        else:
            raise ValueError(f"'level' must be 'patent' or 'claim', got: {level!r}")

    async def _search_patents(
        self,
        dense_vector: List[float],
        top_k: int,
        metadata_filter: Dict[str, Any],
        query_text: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Hybrid patent search.

        Uses Qdrant's native Prefetch + RRF fusion if ``query_text`` is
        provided. Falls back to dense-only when ``query_text`` is absent.
        """
        if query_text:
            return await self.store.search_hybrid(
                query_text=query_text,
                query_dense_vector=dense_vector,
                top_k=top_k,
                metadata_filter=metadata_filter or None,
            )
        else:
            # Dense-only fallback (no BM25 arm) — still the patent collection.
            return await self.store.search_patents_dense(
                query_dense_vector=dense_vector,
                top_k=top_k,
                metadata_filter=metadata_filter or None,
            )

    async def _search_claims(
        self,
        dense_vector: List[float],
        top_k: int,
        metadata_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Dense-only claim search."""
        return await self.store.search_claims_dense(
            query_dense_vector=dense_vector,
            top_k=top_k,
            metadata_filter=metadata_filter or None,
        )
