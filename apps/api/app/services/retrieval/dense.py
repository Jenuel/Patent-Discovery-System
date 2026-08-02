from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.indexing.qdrant import QdrantHybridStore

PATENT_ARMS = ("hybrid", "dense", "bm25")


class DenseRetriever:
    """
    Responsible for dense (and hybrid) retrieval backed entirely by Qdrant.

    Routing by ``level``:
    - ``"patent"`` → :meth:`~QdrantHybridStore.search_hybrid` on ``patents_hybrid``
      (Qdrant-native Prefetch + RRF fusion over dense + BM25 vectors).
    - ``"claim"``  → :meth:`~QdrantHybridStore.search_claims_dense` on ``claims_hybrid``
      (pure dense cosine search, optionally filtered by patent_id).

    Both methods return the unified ``[{id, score, metadata}]`` schema.

    The patent-level arm and its fusion parameters are set **once, at
    construction**, rather than per call. That keeps :meth:`search`'s signature
    stable for every caller (notably ``HierarchicalRetriever``, which needs no
    knowledge of the ablation knobs) while letting the evaluation harness swap
    arms by building a differently-configured retriever and then driving the
    ordinary, unmodified production path.
    """

    def __init__(
        self,
        store: QdrantHybridStore,
        arm: str = "hybrid",
        fusion: str = "rrf",
        prefetch_multiplier: int = 3,
        dense_prefetch_limit: Optional[int] = None,
        sparse_prefetch_limit: Optional[int] = None,
    ):
        """
        Args:
            store:                Qdrant hybrid store.
            arm:                  Patent-level retrieval arm — ``"hybrid"``
                                  (production default), ``"dense"``, or
                                  ``"bm25"``. The three compared in
                                  EVAL_ABLATION.md Test 1.
            fusion:               Server-side fusion for the hybrid arm,
                                  ``"rrf"`` or ``"dbsf"``. Ignored by the
                                  single-arm modes.
            prefetch_multiplier:  Candidates each arm fetches, as a multiple of
                                  ``top_k``. Hybrid arm only.
            dense_prefetch_limit: Absolute dense-arm candidate count, overriding
                                  the multiplier. Hybrid arm only.
            sparse_prefetch_limit: Absolute BM25-arm candidate count. Hybrid
                                  arm only.
        """
        if arm not in PATENT_ARMS:
            raise ValueError(f"arm must be one of {list(PATENT_ARMS)}, got {arm!r}")

        self.store = store
        self.arm = arm
        self.fusion = fusion
        self.prefetch_multiplier = prefetch_multiplier
        self.dense_prefetch_limit = dense_prefetch_limit
        self.sparse_prefetch_limit = sparse_prefetch_limit

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
        Patent search on the arm this retriever was configured with.

        ``"hybrid"`` uses Qdrant's native Prefetch + fusion, falling back to
        dense-only when ``query_text`` is absent — a bare vector gives the BM25
        arm nothing to match on. ``"dense"`` and ``"bm25"`` pin a single arm
        regardless of ``query_text``; the BM25 arm has no vector to fall back
        to, so a term-less query there legitimately returns nothing.
        """
        if self.arm == "dense" or (self.arm == "hybrid" and not query_text):
            return await self.store.search_patents_dense(
                query_dense_vector=dense_vector,
                top_k=top_k,
                metadata_filter=metadata_filter or None,
            )

        if self.arm == "bm25":
            if not query_text:
                raise ValueError("arm='bm25' requires query_text")
            return await self.store.search_bm25(
                query_text=query_text,
                top_k=top_k,
                metadata_filter=metadata_filter or None,
            )

        return await self.store.search_hybrid(
            query_text=query_text,
            query_dense_vector=dense_vector,
            top_k=top_k,
            metadata_filter=metadata_filter or None,
            prefetch_multiplier=self.prefetch_multiplier,
            fusion=self.fusion,
            dense_prefetch_limit=self.dense_prefetch_limit,
            sparse_prefetch_limit=self.sparse_prefetch_limit,
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
