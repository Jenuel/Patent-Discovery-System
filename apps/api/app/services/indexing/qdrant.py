
from __future__ import annotations

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional

import anyio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    Range,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.logging import get_logger
from .schemas import QdrantConfig

log = get_logger(__name__)


def _doc_id_to_uint64(doc_id: str) -> int:
    """Convert an arbitrary string document ID to a uint64 via SHA-256."""
    digest = hashlib.sha256(doc_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


# BM25 encoding is delegated to fastembed's maintained "Qdrant/bm25" model:
# deterministic term hashing, stemming, and stopword removal, with document
# values carrying only the TF-saturation component — the IDF factor is applied
# server-side by Qdrant (collection uses ``Modifier.IDF``).
_BM25_MODEL_NAME = "Qdrant/bm25"

_bm25_encoder: Optional[Any] = None
_bm25_encoder_lock = threading.Lock()


def _get_bm25_encoder() -> Any:
    """
    Return the process-wide BM25 encoder, creating it on first use.

    ``fastembed`` is imported lazily so that importing this module (for the
    store's filter/collection logic) does not drag in the ONNX stack.

    Construction downloads the model artifacts, so it is guarded by a lock:
    encoding now runs on worker threads and two concurrent first calls would
    otherwise each build (and download) their own instance.
    """
    global _bm25_encoder
    if _bm25_encoder is None:
        with _bm25_encoder_lock:
            if _bm25_encoder is None:
                from fastembed import SparseTextEmbedding

                _bm25_encoder = SparseTextEmbedding(model_name=_BM25_MODEL_NAME)
    return _bm25_encoder


async def warm_bm25_encoder() -> None:
    """
    Build the BM25 encoder off the event loop.

    Call once at application startup so the first query does not pay the model
    download/initialisation cost while blocking the loop.
    """
    await anyio.to_thread.run_sync(_get_bm25_encoder)


async def _encode_sparse_documents(texts: List[str]) -> List[Any]:
    """Encode document texts to BM25 sparse embeddings on a worker thread."""
    return await anyio.to_thread.run_sync(
        lambda: list(_get_bm25_encoder().embed(texts))
    )


async def _encode_sparse_query(query_text: str) -> Optional[SparseVector]:
    """Encode a query to a BM25 sparse vector on a worker thread.

    Uses ``query_embed`` (not ``embed``): query-side BM25 values carry no
    TF saturation, matching the IDF-only weighting Qdrant applies server-side.

    Returns *None* for a query that yields no terms at all (e.g. entirely
    stopwords), which callers must treat as "no BM25 arm available".
    """
    embedding = await anyio.to_thread.run_sync(
        lambda: next(iter(_get_bm25_encoder().query_embed(query_text)))
    )
    return _to_sparse_vector(embedding)


def _to_sparse_vector(embedding: Any) -> Optional[SparseVector]:
    """
    Convert a fastembed SparseEmbedding to a Qdrant SparseVector.

    Returns *None* when the text produced no terms — stopwords only, or empty.

    It is important that this is *None* rather than a placeholder vector. The
    collection uses ``Modifier.IDF``, so Qdrant derives inverse document
    frequency from collection-wide document counts: seeding every term-less
    document with the same dummy token would inflate that token's document
    frequency and perturb the scores of every other query. Named vectors are
    optional per point, so such documents are simply indexed dense-only.
    """
    indices = embedding.indices.tolist()
    values = embedding.values.tolist()
    if not indices:
        return None
    return SparseVector(indices=indices, values=values)


class QdrantHybridStore:
    """
    Qdrant wrapper that manages two hybrid-vector collections:

    ``patents_hybrid``
        Patent-level points — each point stores:
          - ``dense``: OpenAI dense embedding (e.g. text-embedding-3-small, 1536-dim)
          - ``bm25``:  BM25 sparse vector (fastembed ``Qdrant/bm25``, IDF applied server-side)

        Supports native hybrid search via Qdrant Prefetch + RRF fusion.

    ``claims_hybrid``
        Claim-level points — each point stores:
          - ``dense``: OpenAI dense embedding only (no sparse at claim level)

        Supports dense filtered search by patent_id.
    """

    SPARSE_VECTOR_NAME = "bm25"
    DENSE_VECTOR_NAME = "dense"

    PATENT_PAYLOAD_INDEXES = {
        "patent_id": PayloadSchemaType.KEYWORD,
        "cpc": PayloadSchemaType.KEYWORD,
        "cpc_prefix": PayloadSchemaType.KEYWORD,
        "year": PayloadSchemaType.INTEGER,
    }

    CLAIM_PAYLOAD_INDEXES = {
        "patent_id": PayloadSchemaType.KEYWORD,
    }

    def __init__(
        self,
        cfg: QdrantConfig,
        client: Optional[AsyncQdrantClient] = None,
    ):
        """
        Args:
            cfg:    Connection and collection configuration.
            client: Pre-built client, mainly for tests. When omitted a real
                    ``AsyncQdrantClient`` is constructed, which probes the
                    server for a version-compatibility check.
        """
        if not cfg.url:
            raise ValueError("QDRANT_URL is required for Qdrant")

        self.cfg = cfg
        self._client = client or AsyncQdrantClient(
            url=cfg.url,
            api_key=cfg.api_key or None,
            timeout=cfg.timeout,
        )

    @classmethod
    def from_env(cls) -> "QdrantHybridStore":
        """Create QdrantHybridStore from environment variables."""
        return cls(
            QdrantConfig(
                url=os.getenv("QDRANT_URL", ""),
                api_key=os.getenv("QDRANT_API_KEY", ""),
                patent_collection_name=os.getenv(
                    "QDRANT_PATENT_COLLECTION_NAME", "patents_hybrid"
                ),
                claim_collection_name=os.getenv(
                    "QDRANT_CLAIM_COLLECTION_NAME", "claims_hybrid"
                ),
                dense_vector_size=int(os.getenv("QDRANT_DENSE_VECTOR_SIZE", "1536")),
                timeout=float(os.getenv("QDRANT_TIMEOUT", "30")),
                upload_batch_size=int(os.getenv("QDRANT_UPLOAD_BATCH_SIZE", "100")),
            )
        )

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def create_collections(self) -> None:
        """
        Create both Qdrant collections if they do not already exist.

        ``patents_hybrid``  — dense + BM25 sparse vectors.
        ``claims_hybrid``   — dense vector only.

        Also ensures each collection's payload indexes exist.

        Safe to call repeatedly: existing collections and existing payload
        indexes are left alone. Failures are *not* swallowed — a collection or
        index that could not be created is a setup error worth surfacing.
        """
        await self._create_patent_collection()
        await self._create_claim_collection()
        await self._create_payload_indexes()

    async def _create_payload_indexes(self) -> None:
        """
        Create the payload indexes each collection's filters need.

        Existing indexes are detected by reading ``CollectionInfo.payload_schema``
        rather than by attempting creation and swallowing the error. Failures
        therefore propagate: a missing payload index is invisible until Stage 2
        gets slow, which is exactly the kind of problem that must not be
        downgraded to a debug log.
        """
        for collection, wanted in (
            (self.cfg.patent_collection_name, self.PATENT_PAYLOAD_INDEXES),
            (self.cfg.claim_collection_name, self.CLAIM_PAYLOAD_INDEXES),
        ):
            info = await self._client.get_collection(collection)
            existing = set(info.payload_schema or {})

            for field, schema in wanted.items():
                if field in existing:
                    log.debug(
                        f"[QDRANT] Payload index already present on '{collection}.{field}'"
                    )
                    continue

                await self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )
                log.info(f"[QDRANT] Payload index created on '{collection}.{field}'")

    async def _create_patent_collection(self) -> None:
        """Create the patents_hybrid collection (dense + sparse)."""
        name = self.cfg.patent_collection_name
        if await self._collection_exists(name):
            log.info(f"[QDRANT] Collection '{name}' already exists — skipping creation")
            return

        log.info(f"[QDRANT] Creating patent collection '{name}'")
        await self._client.create_collection(
            collection_name=name,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=self.cfg.dense_vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    modifier=Modifier.IDF,
                )
            },
        )
        log.info(f"[QDRANT] Patent collection '{name}' created successfully")

    async def _create_claim_collection(self) -> None:
        """Create the claims_hybrid collection (dense only)."""
        name = self.cfg.claim_collection_name
        if await self._collection_exists(name):
            log.info(f"[QDRANT] Collection '{name}' already exists — skipping creation")
            return

        log.info(f"[QDRANT] Creating claim collection '{name}'")
        await self._client.create_collection(
            collection_name=name,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=self.cfg.dense_vector_size,
                    distance=Distance.COSINE,
                )
            },
            # No sparse vectors at claim level
            sparse_vectors_config={},
        )
        log.info(f"[QDRANT] Claim collection '{name}' created successfully")

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        """
        True only for a genuine "collection does not exist" error.

        When the client gives us a status code we trust it exclusively — a 401
        whose body happens to contain the words "not found" must not be read as
        absence.
        """
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status == 404
        return "not found" in str(exc).lower()

    async def _collection_exists(self, name: str) -> bool:
        """
        Return True if the named collection already exists in Qdrant.

        Only a genuine "not found" counts as absence — auth and network errors
        are re-raised so a bad API key surfaces as itself rather than as a
        confusing "collection missing, creating…" followed by a second failure.
        """
        try:
            await self._client.get_collection(name)
            return True
        except Exception as exc:
            if self._is_not_found(exc):
                return False
            raise

    async def delete_collections(self) -> None:
        """
        Delete both managed collections.

        An already-absent collection is fine; **any other failure raises.**
        Swallowing them is not safe here: this is the ``--recreate`` path, and a
        delete that silently fails leaves the collection in place, so the
        subsequent ``create_collections`` skips creation and indexing upserts
        into the *old* collection. Points from a previous corpus that are no
        longer in the input survive, and a full re-index quietly degrades into
        an incremental update.
        """
        for name in (self.cfg.patent_collection_name, self.cfg.claim_collection_name):
            log.info(f"[QDRANT] Deleting collection '{name}'")
            try:
                await self._client.delete_collection(name)
                log.info(f"[QDRANT] Collection '{name}' deleted")
            except Exception as exc:
                if self._is_not_found(exc):
                    log.info(f"[QDRANT] Collection '{name}' did not exist — nothing to delete")
                    continue
                raise

    # ------------------------------------------------------------------
    # Indexing — patent level
    # ------------------------------------------------------------------

    async def bulk_index_patents(
        self,
        documents: List[Dict[str, Any]],
        dense_vectors: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Bulk-index patent documents into ``patents_hybrid``.

        Each point stores both a BM25 sparse vector and a dense vector.

        Args:
            documents:     List of documents, each must contain ``_id`` plus
                           patent payload fields (title, abstract, claims, …).
            dense_vectors: Pre-computed OpenAI dense embeddings aligned with
                           ``documents``. If *None*, embeddings are computed
                           internally using :func:`app.services.indexing.embed.embed_texts`.
        """
        log.info(f"[QDRANT] Bulk indexing {len(documents)} patent documents")

        # Step 1 — BM25 sparse embeddings (title-weighted text)
        sparse_texts = [self._extract_patent_text_sparse(doc) for doc in documents]
        sparse_embeddings = await _encode_sparse_documents(sparse_texts)

        # Step 2 — Dense embeddings (natural prose — see _extract_patent_text_dense)
        if dense_vectors is None:
            from app.services.indexing.embed import embed_texts

            dense_texts = [self._extract_patent_text_dense(doc) for doc in documents]
            log.info("[QDRANT] Computing dense embeddings for patent documents…")
            dense_vectors = await anyio.to_thread.run_sync(
                lambda: embed_texts(dense_texts)
            )

        if len(dense_vectors) != len(documents):
            raise ValueError(
                f"dense_vectors length ({len(dense_vectors)}) does not match "
                f"documents length ({len(documents)})"
            )

        # Step 3 — Build Qdrant points
        points: List[PointStruct] = []
        no_sparse = 0
        for doc, sparse_emb, dense_vec in zip(documents, sparse_embeddings, dense_vectors):
            doc_id = doc["_id"]
            payload = {k: v for k, v in doc.items() if k != "_id"}
            payload["_doc_id"] = doc_id

            # Named vectors are optional per point: a document with no BM25
            # terms is indexed dense-only rather than with a dummy sparse
            # vector that would skew the collection's IDF statistics.
            vectors: Dict[str, Any] = {self.DENSE_VECTOR_NAME: dense_vec}
            sparse_vec = _to_sparse_vector(sparse_emb)
            if sparse_vec is not None:
                vectors[self.SPARSE_VECTOR_NAME] = sparse_vec
            else:
                no_sparse += 1

            points.append(
                PointStruct(
                    id=_doc_id_to_uint64(doc_id),
                    payload=payload,
                    vector=vectors,
                )
            )

        if no_sparse:
            log.warning(
                f"[QDRANT] {no_sparse} patent document(s) produced no BM25 terms — "
                f"indexed dense-only (not reachable via the sparse arm)"
            )

        # Step 4 — Upsert in batches
        await self._upsert_batched(points, self.cfg.patent_collection_name)
        log.info(f"[QDRANT] Patent bulk index complete: {len(documents)} documents")

    # ------------------------------------------------------------------
    # Indexing — claim level
    # ------------------------------------------------------------------

    async def bulk_index_claims(
        self,
        documents: List[Dict[str, Any]],
        dense_vectors: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Bulk-index claim-level chunk documents into ``claims_hybrid``.

        Each point stores a dense vector only (no BM25 at claim level).

        Args:
            documents:     List of documents, each must contain ``_id`` plus
                           payload fields (patent_id, section, text, …).
            dense_vectors: Pre-computed OpenAI dense embeddings aligned with
                           ``documents``. If *None*, embeddings are computed
                           internally.
        """
        log.info(f"[QDRANT] Bulk indexing {len(documents)} claim documents")

        # Step 1 — Dense embeddings (compute if not supplied)
        if dense_vectors is None:
            from app.services.indexing.embed import embed_texts

            texts = [self._extract_claim_text(doc) for doc in documents]
            log.info("[QDRANT] Computing dense embeddings for claim documents…")
            dense_vectors = await anyio.to_thread.run_sync(lambda: embed_texts(texts))

        if len(dense_vectors) != len(documents):
            raise ValueError(
                f"dense_vectors length ({len(dense_vectors)}) does not match "
                f"documents length ({len(documents)})"
            )

        # Step 2 — Build Qdrant points
        points: List[PointStruct] = []
        for doc, dense_vec in zip(documents, dense_vectors):
            doc_id = doc["_id"]
            payload = {k: v for k, v in doc.items() if k != "_id"}
            payload["_doc_id"] = doc_id

            points.append(
                PointStruct(
                    id=_doc_id_to_uint64(doc_id),
                    payload=payload,
                    vector={self.DENSE_VECTOR_NAME: dense_vec},
                )
            )

        # Step 3 — Upsert in batches
        await self._upsert_batched(points, self.cfg.claim_collection_name)
        log.info(f"[QDRANT] Claim bulk index complete: {len(documents)} documents")

    async def _upsert_batched(
        self, points: List[PointStruct], collection_name: str
    ) -> None:
        """Upsert a list of points in configured batch sizes."""
        batch_size = self.cfg.upload_batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            log.debug(
                f"[QDRANT] Uploaded batch {i // batch_size + 1} "
                f"({len(batch)} points) to '{collection_name}'"
            )

    # ------------------------------------------------------------------
    # Search — hybrid patent-level (Qdrant native RRF)
    # ------------------------------------------------------------------

    async def search_hybrid(
        self,
        query_text: str,
        query_dense_vector: List[float],
        *,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        prefetch_multiplier: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid patent search using Qdrant's native Prefetch + RRF fusion.

        Issues two prefetch sub-queries simultaneously:
          1. BM25 sparse retrieval (``bm25`` named vector)
          2. Dense cosine retrieval (``dense`` named vector)

        Qdrant fuses both ranked lists with Reciprocal Rank Fusion (RRF)
        server-side, so no Python-level fusion is needed.

        Args:
            query_text:          Raw query string (encoded to BM25 sparse vector).
            query_dense_vector:  Pre-computed OpenAI dense query embedding.
            top_k:               Number of results to return.
            metadata_filter:     Optional payload filters.
            prefetch_multiplier: Each arm fetches ``top_k * multiplier`` candidates
                                 so RRF has room to reorder. With both arms limited
                                 to ``top_k``, a document ranked just outside one
                                 arm's cut-off can never be rescued by the other.

        Returns:
            List of dicts with ``id``, ``score``, and ``metadata``.
        """
        if not query_text:
            raise ValueError("query_text is required")
        if not query_dense_vector:
            raise ValueError("query_dense_vector is required")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        log.info(f"[QDRANT] Hybrid patent search (top_k={top_k})")

        query_sparse_vec = await _encode_sparse_query(query_text)

        if query_sparse_vec is None:
            log.warning(
                "[QDRANT] Query produced no BM25 terms — falling back to dense-only search"
            )
            return await self.search_patents_dense(
                query_dense_vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

        # Build payload filter
        qdrant_filter: Optional[Filter] = None
        if metadata_filter:
            qdrant_filter = self._build_filter(metadata_filter)

        prefetch_limit = max(top_k, top_k * prefetch_multiplier)

        results = await self._client.query_points(
            collection_name=self.cfg.patent_collection_name,
            prefetch=[
                Prefetch(
                    query=query_sparse_vec,
                    using=self.SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=query_dense_vector,
                    using=self.DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        output = self._normalise_results(results.points)
        log.info(f"[QDRANT] Hybrid search complete: {len(output)} results")
        if output:
            log.debug(f"[QDRANT] Top result score: {output[0]['score']:.4f}")
        return output

    # ------------------------------------------------------------------
    # Search — dense-only
    # ------------------------------------------------------------------

    async def search_claims_dense(
        self,
        query_dense_vector: List[float],
        *,
        top_k: int = 30,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense claim search on the ``claims_hybrid`` collection.

        Typically called with a ``patent_id: {$in: [...]}`` filter to restrict
        results to claims belonging to patents selected in Stage 1.

        Args:
            query_dense_vector: Pre-computed OpenAI dense query embedding.
            top_k:              Number of results to return.
            metadata_filter:    Optional payload filters (supports equality,
                                range, and ``$in`` operators).

        Returns:
            List of dicts with ``id``, ``score``, and ``metadata``.
        """
        return await self._search_dense(
            self.cfg.claim_collection_name,
            query_dense_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            label="claim",
        )

    async def search_patents_dense(
        self,
        query_dense_vector: List[float],
        *,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense-only patent search on the ``patents_hybrid`` collection.

        Fallback for when no query text is available to drive the BM25 arm.
        Prefer :meth:`search_hybrid` for normal use.
        """
        return await self._search_dense(
            self.cfg.patent_collection_name,
            query_dense_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            label="patent",
        )

    async def _search_dense(
        self,
        collection_name: str,
        query_dense_vector: List[float],
        *,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]],
        label: str,
    ) -> List[Dict[str, Any]]:
        """Dense cosine search against a named collection."""
        if not query_dense_vector:
            raise ValueError("query_dense_vector is required")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        log.info(f"[QDRANT] Dense {label} search (top_k={top_k})")

        qdrant_filter: Optional[Filter] = None
        if metadata_filter:
            qdrant_filter = self._build_filter(metadata_filter)

        results = await self._client.query_points(
            collection_name=collection_name,
            query=query_dense_vector,
            using=self.DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        output = self._normalise_results(results.points)
        log.info(f"[QDRANT] Dense {label} search complete: {len(output)} results")
        if output:
            log.debug(f"[QDRANT] Top result score: {output[0]['score']:.4f}")
        return output

    # ------------------------------------------------------------------
    # Search — BM25 sparse (backward compatible, patent-level only)
    # ------------------------------------------------------------------

    async def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_fields: Optional[List[str]] = None,  # kept for API compatibility
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 sparse retrieval on ``patents_hybrid``.

        Kept for backward compatibility and as a fallback when no dense
        query vector is available. Prefer :meth:`search_hybrid` for normal use.

        Args:
            query_text:     Query text for BM25 search.
            top_k:          Number of results to return.
            metadata_filter: Optional metadata filters.
            search_fields:  Ignored — kept for API compatibility with old
                            ElasticsearchStore signature.

        Returns:
            List of dicts with ``id``, ``score``, and ``metadata``.
        """
        if not query_text:
            raise ValueError("query_text is required")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        log.info(f"[QDRANT] BM25 sparse search (top_k={top_k})")
        log.debug(f"[QDRANT] Query text: '{query_text[:100]}...'")

        query_vec = await _encode_sparse_query(query_text)
        if query_vec is None:
            # No BM25 terms and, unlike search_hybrid, no dense vector to fall
            # back to. An empty result is the honest answer.
            log.warning("[QDRANT] Query produced no BM25 terms — returning no results")
            return []

        qdrant_filter: Optional[Filter] = None
        if metadata_filter:
            log.debug(f"[QDRANT] Applying filters: {metadata_filter}")
            qdrant_filter = self._build_filter(metadata_filter)

        results = await self._client.query_points(
            collection_name=self.cfg.patent_collection_name,
            query=query_vec,
            using=self.SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        output = self._normalise_results(results.points)
        log.info(f"[QDRANT] BM25 search complete: {len(output)} results")
        if output:
            log.debug(f"[QDRANT] Top result score: {output[0]['score']:.4f}")
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_results(self, scored_points: list) -> List[Dict[str, Any]]:
        """Convert Qdrant ScoredPoint list to the unified result schema."""
        output: List[Dict[str, Any]] = []
        for sp in scored_points:
            payload = dict(sp.payload or {})
            doc_id = payload.pop("_doc_id", str(sp.id))
            output.append(
                {
                    "id": doc_id,
                    "score": sp.score,
                    "metadata": payload,
                }
            )
        return output

    def _build_filter(self, metadata_filter: Dict[str, Any]) -> Filter:
        """
        Convert a metadata filter dict (Elasticsearch-style operators) to a
        Qdrant ``Filter`` object.

        Supported operators:
          - Simple equality:  ``{\"field\": value}``
          - Explicit equality:``{\"field\": {\"$eq\": value}}``
          - Range:            ``{\"field\": {\"$gte\": v, \"$lte\": v, \"$gt\": v, \"$lt\": v}}``
          - In-list:          ``{\"field\": {\"$in\": [v1, v2, ...]}}``

        Unrecognised operators raise rather than being dropped: a silently
        discarded condition widens the search instead of narrowing it, which
        is exactly backwards for a filter (Stage 2's ``patent_id`` allowlist
        would stop restricting anything).
        """
        must_conditions = []

        for key, value in metadata_filter.items():
            if isinstance(value, dict):
                range_ops = {"$gte": "gte", "$lte": "lte", "$gt": "gt", "$lt": "lt"}
                if any(op in value for op in range_ops):
                    unknown = set(value) - set(range_ops)
                    if unknown:
                        raise ValueError(
                            f"Unsupported operator(s) {sorted(unknown)} alongside a "
                            f"range filter on '{key}'"
                        )
                    range_kwargs = {
                        dest: value[op] for op, dest in range_ops.items() if op in value
                    }
                    must_conditions.append(
                        FieldCondition(key=key, range=Range(**range_kwargs))
                    )
                elif "$in" in value:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value["$in"]))
                    )
                elif "$eq" in value:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value["$eq"]))
                    )
                else:
                    raise ValueError(
                        f"Unsupported filter operator(s) {sorted(value)} on '{key}'. "
                        f"Supported: $eq, $in, $gt, $gte, $lt, $lte"
                    )
            else:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=must_conditions)

    @staticmethod
    def _extract_patent_text_sparse(document: Dict[str, Any]) -> str:
        """
        Build the BM25 input for a patent document.

        Title is repeated to raise its term frequency — crude field weighting,
        but that is how you weight a field under BM25. ``patent_id`` is included
        so a query naming a publication number matches it lexically.
        """
        parts: List[str] = []
        title = document.get("title", "")
        if title:
            parts.extend([title, title])
        for field in ("abstract", "text", "claims", "patent_id"):
            val = document.get(field)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    @staticmethod
    def _extract_patent_text_dense(document: Dict[str, Any]) -> str:
        """
        Build the dense-embedding input for a patent document.

        Differs from the BM25 input in two ways, both deliberate:
          - the title appears **once** — repeating it skews the embedding
            toward the title and away from the abstract and claims;
          - ``patent_id`` is **excluded** — an alphanumeric publication number
            carries no semantic signal and only adds noise tokens.
        """
        parts: List[str] = []
        for field in ("title", "abstract", "text", "claims"):
            val = document.get(field)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    @staticmethod
    def _extract_claim_text(document: Dict[str, Any]) -> str:
        """Extract text content from a claim-level chunk document."""
        return (
            document.get("text")
            or document.get("raw_text")
            or document.get("content")
            or ""
        )

QdrantSparseStore = QdrantHybridStore
