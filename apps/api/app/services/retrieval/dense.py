from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import anyio

from app.services.indexing.qdrant import QdrantHybridStore
from app.services.retrieval.fusion import (
    DEFAULT_RRF_K,
    to_result_dicts,
    to_scored_matches,
    weighted_rrf,
)

PATENT_ARMS = ("hybrid", "dense", "bm25", "weighted")


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
        dense_weight: float = 0.9,
        sparse_weight: float = 0.1,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        """
        Args:
            store:                Qdrant hybrid store.
            arm:                  Patent-level retrieval arm — ``"hybrid"``
                                  (production default), ``"dense"``, ``"bm25"``,
                                  or ``"weighted"``. The first three are
                                  compared in EVAL_ABLATION.md Test 1;
                                  ``"weighted"`` is Test 6 (RET-09).
            fusion:               Server-side fusion for the hybrid arm,
                                  ``"rrf"`` or ``"dbsf"``. Ignored by the
                                  single-arm modes and by ``"weighted"``, which
                                  fuses client-side.
            prefetch_multiplier:  Candidates each arm fetches, as a multiple of
                                  ``top_k``. Hybrid and weighted arms.
            dense_prefetch_limit: Absolute dense-arm candidate count, overriding
                                  the multiplier. Hybrid and weighted arms.
            sparse_prefetch_limit: Absolute BM25-arm candidate count. Hybrid and
                                  weighted arms.
            dense_weight:         Dense-arm weight for ``arm="weighted"``.
                                  Defaults to the 9:1 ratio EVAL_ABLATION.md
                                  Test 6 identified as the point where the
                                  recall curve flattens.
            sparse_weight:        BM25-arm weight for ``arm="weighted"``.
            rrf_k:                RRF smoothing constant for ``arm="weighted"``.
                                  Qdrant's native RRF k is fixed server-side, so
                                  this applies to client-side fusion only.

        Raises:
            ValueError: on an unknown ``arm``, a negative weight, or a weighted
                arm with both weights zero.
        """
        if arm not in PATENT_ARMS:
            raise ValueError(f"arm must be one of {list(PATENT_ARMS)}, got {arm!r}")

        for name, weight in (("dense_weight", dense_weight), ("sparse_weight", sparse_weight)):
            if weight < 0:
                raise ValueError(f"{name} must be >= 0, got {weight}")

        if arm == "weighted" and not (dense_weight or sparse_weight):
            raise ValueError(
                "arm='weighted' needs a non-zero dense_weight or sparse_weight; "
                "both zero retrieves nothing. Use arm='dense' or arm='bm25' to "
                "pin a single arm."
            )

        self.store = store
        self.arm = arm
        self.fusion = fusion
        self.prefetch_multiplier = prefetch_multiplier
        self.dense_prefetch_limit = dense_prefetch_limit
        self.sparse_prefetch_limit = sparse_prefetch_limit
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

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

        ``"hybrid"`` uses Qdrant's native Prefetch + fusion and ``"weighted"``
        fuses the same two arms client-side; both fall back to dense-only when
        ``query_text`` is absent — a bare vector gives the BM25 arm nothing to
        match on. ``"dense"`` and ``"bm25"`` pin a single arm regardless of
        ``query_text``; the BM25 arm has no vector to fall back to, so a
        term-less query there legitimately returns nothing.
        """
        text_dependent = self.arm in ("hybrid", "weighted")
        if self.arm == "dense" or (text_dependent and not query_text):
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

        if self.arm == "weighted":
            return await self._search_weighted(
                dense_vector=dense_vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                query_text=query_text,
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

    def _prefetch_limits(self, top_k: int) -> Tuple[int, int]:
        """Per-arm candidate counts as ``(dense_limit, sparse_limit)``.

        Deliberately mirrors ``QdrantHybridStore.search_hybrid``'s sizing so the
        weighted arm fuses over the *same* candidate pool as the native one —
        otherwise a weighted-vs-native comparison would be confounded by depth
        as well as by weight. The duplication is preferred over a shared helper
        because ``qdrant.py`` lives in the ``indexing`` package and this module
        already depends on it; sharing would invert that direction for three
        lines of arithmetic.
        """
        prefetch_limit = max(top_k, top_k * self.prefetch_multiplier)
        dense_limit = (
            self.dense_prefetch_limit if self.dense_prefetch_limit is not None else prefetch_limit
        )
        sparse_limit = (
            self.sparse_prefetch_limit if self.sparse_prefetch_limit is not None else prefetch_limit
        )
        return dense_limit, sparse_limit

    async def _search_weighted(
        self,
        dense_vector: List[float],
        top_k: int,
        metadata_filter: Dict[str, Any],
        query_text: str,
    ) -> List[Dict[str, Any]]:
        """Client-side weighted RRF over the dense and BM25 arms (RET-09).

        Qdrant's ``FusionQuery`` takes no weight parameter, so the two arms are
        fetched independently and fused here — see
        :func:`~app.services.retrieval.fusion.weighted_rrf`. This costs two
        round trips against the hybrid arm's one, so they are issued
        concurrently; the added latency is one arm's, not both.

        A failure in either arm propagates. Retrieval is Stage 1's whole job,
        and ``HierarchicalRetriever`` already catches and degrades to empty
        results at the stage boundary — swallowing it here would silently
        return a single-arm ranking under a weighted label instead.
        """
        dense_limit, sparse_limit = self._prefetch_limits(top_k)
        arm_filter = metadata_filter or None
        dense_raw: List[Dict[str, Any]] = []
        sparse_raw: List[Dict[str, Any]] = []

        async def _dense() -> None:
            nonlocal dense_raw
            dense_raw = await self.store.search_patents_dense(
                query_dense_vector=dense_vector,
                top_k=dense_limit,
                metadata_filter=arm_filter,
            )

        async def _sparse() -> None:
            nonlocal sparse_raw
            sparse_raw = await self.store.search_bm25(
                query_text=query_text,
                top_k=sparse_limit,
                metadata_filter=arm_filter,
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(_dense)
            tg.start_soon(_sparse)

        fused = weighted_rrf(
            [
                (self.dense_weight, to_scored_matches(dense_raw)),
                (self.sparse_weight, to_scored_matches(sparse_raw)),
            ],
            k=self.rrf_k,
            limit=top_k,
        )
        return to_result_dicts(fused)

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
