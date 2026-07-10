
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
)

from app.core.logging import get_logger
from .schemas import QdrantConfig

log = get_logger(__name__)

def _doc_id_to_uint64(doc_id: str) -> int:
    """Convert an arbitrary string document ID to a uint64 via SHA-256."""
    digest = hashlib.sha256(doc_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _tokenise(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser (lowercase)."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_bm25_sparse_vector(
    tokens: List[str],
    idf: Dict[str, float],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 100.0,
    dl: float = 100.0,
) -> SparseVector:
    """
    Compute a BM25 sparse vector for a list of tokens.

    Returns a ``SparseVector`` ready for Qdrant upload.
    Each unique token maps to an integer index (stable hash) and the
    corresponding BM25-TF weight is the value.
    """
    from collections import Counter

    tf = Counter(tokens)
    indices: List[int] = []
    values: List[float] = []

    for term, freq in tf.items():
        term_idx = abs(hash(term)) % (2**24)  
        term_idf = idf.get(term, 0.1)  

        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * dl / avg_dl)
        bm25_weight = term_idf * (numerator / denominator)

        indices.append(term_idx)
        values.append(float(bm25_weight))

    if not indices:
        indices = [0]
        values = [0.0]

    return SparseVector(indices=indices, values=values)


class QdrantSparseStore:
    """
    Qdrant wrapper for BM25 sparse retrieval.
    Used for patent-level lexical search (mirrors the old ElasticsearchStore API).
    """
    SPARSE_VECTOR_NAME = "bm25"

    def __init__(self, cfg: QdrantConfig):
        if not cfg.url and not cfg.api_key:
            raise ValueError("Either url or api_key must be provided for Qdrant")

        self.cfg = cfg

        # Build async client
        self._client = AsyncQdrantClient(
            url=cfg.url or "https://localhost:6333",
            api_key=cfg.api_key or None,
            timeout=cfg.timeout,
        )

        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 100.0

    @classmethod
    def from_env(cls) -> "QdrantSparseStore":
        """Create QdrantSparseStore from environment variables."""
        return cls(
            QdrantConfig(
                url=os.getenv("QDRANT_URL", ""),
                api_key=os.getenv("QDRANT_API_KEY", ""),
                collection_name=os.getenv("QDRANT_COLLECTION_NAME", "patents_sparse"),
                timeout=float(os.getenv("QDRANT_TIMEOUT", "30")),
            )
        )


    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self._client.close()


    async def create_collection(self) -> None:
        """
        Create the Qdrant collection with a sparse vector field for BM25.
        Safe to call if the collection already exists (no-op).
        """
        existing = await self.collection_exists()
        if existing:
            log.info(
                f"[QDRANT] Collection '{self.cfg.collection_name}' already exists — skipping creation"
            )
            return

        log.info(f"[QDRANT] Creating collection '{self.cfg.collection_name}'")
        await self._client.create_collection(
            collection_name=self.cfg.collection_name,
            # No dense vectors — this collection is sparse-only
            vectors_config={},
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        log.info(f"[QDRANT] Collection '{self.cfg.collection_name}' created successfully")

    async def delete_collection(self) -> None:
        """Delete the Qdrant collection."""
        log.info(f"[QDRANT] Deleting collection '{self.cfg.collection_name}'")
        await self._client.delete_collection(self.cfg.collection_name)
        log.info(f"[QDRANT] Collection '{self.cfg.collection_name}' deleted")

    async def collection_exists(self) -> bool:
        """Return True if the collection already exists in Qdrant."""
        try:
            await self._client.get_collection(self.cfg.collection_name)
            return True
        except Exception:
            return False


    def _update_idf(self, corpus_tokens: List[List[str]]) -> None:
        """
        Compute IDF values from a corpus of tokenised documents and update
        the in-memory IDF table.
        """
        import math
        from collections import Counter

        N = len(corpus_tokens)
        df: Dict[str, int] = Counter()
        total_len = 0

        for tokens in corpus_tokens:
            total_len += len(tokens)
            for term in set(tokens):
                df[term] += 1

        if N:
            self._avg_dl = total_len / N

        for term, freq in df.items():
            self._idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)


    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> None:
        """
        Index a single document as a Qdrant sparse-vector point.

        Args:
            doc_id: String document ID (hashed to uint64 for Qdrant)
            document: Document payload (patent fields + text)
        """
        log.debug(f"[QDRANT] Indexing document: {doc_id}")

        text = self._extract_text(document)
        tokens = _tokenise(text)

        # Build a minimal IDF if not populated yet
        if not self._idf:
            self._update_idf([tokens])

        sparse_vec = _compute_bm25_sparse_vector(
            tokens, self._idf, avg_dl=self._avg_dl, dl=len(tokens)
        )

        point = PointStruct(
            id=_doc_id_to_uint64(doc_id),
            payload={**document, "_doc_id": doc_id},
            vector={self.SPARSE_VECTOR_NAME: sparse_vec},
        )

        await self._client.upsert(
            collection_name=self.cfg.collection_name,
            points=[point],
        )
        log.debug(f"[QDRANT] Document indexed: {doc_id}")

    async def bulk_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Bulk index documents into Qdrant.

        Args:
            documents: List of documents, each must contain a ``_id`` field
                       plus patent payload fields.
        """
        log.info(f"[QDRANT] Bulk indexing {len(documents)} documents")

        # Step 1 — build corpus for IDF estimation
        corpus_tokens: List[List[str]] = []
        for doc in documents:
            text = self._extract_text(doc)
            corpus_tokens.append(_tokenise(text))

        self._update_idf(corpus_tokens)
        log.debug(
            f"[QDRANT] IDF table built ({len(self._idf)} unique terms, avg_dl={self._avg_dl:.1f})"
        )

        # Step 2 — build Qdrant points
        points: List[PointStruct] = []
        for doc, tokens in zip(documents, corpus_tokens):
            doc_id = doc["_id"]
            payload = {k: v for k, v in doc.items() if k != "_id"}
            payload["_doc_id"] = doc_id

            sparse_vec = _compute_bm25_sparse_vector(
                tokens, self._idf, avg_dl=self._avg_dl, dl=len(tokens)
            )

            points.append(
                PointStruct(
                    id=_doc_id_to_uint64(doc_id),
                    payload=payload,
                    vector={self.SPARSE_VECTOR_NAME: sparse_vec},
                )
            )

        # Step 3 — upsert in batches
        batch_size = self.cfg.upload_batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=self.cfg.collection_name,
                points=batch,
            )
            log.debug(
                f"[QDRANT] Uploaded batch {i // batch_size + 1} ({len(batch)} points)"
            )

        log.info(f"[QDRANT] Bulk index complete: {len(documents)} documents")


    async def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_fields: Optional[List[str]] = None,  # kept for API compatibility
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 sparse retrieval using Qdrant sparse vectors.

        Args:
            query_text: Query text for BM25 search
            top_k: Number of results to return
            metadata_filter: Optional metadata filters (same schema as ElasticsearchStore)
            search_fields: Ignored — Qdrant searches the full BM25 vector; kept for
                           API compatibility with the old ElasticsearchStore signature.

        Returns:
            List of search results, each a dict with ``id``, ``score``, and ``metadata``.
        """
        if not query_text:
            raise ValueError("query_text is required")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        log.info(f"[QDRANT] BM25 sparse search (top_k={top_k})")
        log.debug(f"[QDRANT] Query text: '{query_text[:100]}...'")

        # Tokenise query and build its sparse vector
        tokens = _tokenise(query_text)
        query_vec = _compute_bm25_sparse_vector(
            tokens, self._idf, avg_dl=self._avg_dl, dl=len(tokens)
        )

        # Build Qdrant filter from metadata_filter dict
        qdrant_filter: Optional[Filter] = None
        if metadata_filter:
            log.debug(f"[QDRANT] Applying filters: {metadata_filter}")
            qdrant_filter = self._build_filter(metadata_filter)

        # Execute sparse query
        results = await self._client.query_points(
            collection_name=self.cfg.collection_name,
            query=query_vec,
            using=self.SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Normalise output to match old ElasticsearchStore schema
        output: List[Dict[str, Any]] = []
        for scored_point in results.points:
            payload = dict(scored_point.payload or {})
            doc_id = payload.pop("_doc_id", str(scored_point.id))
            output.append(
                {
                    "id": doc_id,
                    "score": scored_point.score,
                    "metadata": payload,
                }
            )

        log.info(f"[QDRANT] BM25 search complete: found {len(output)} results")
        if output:
            log.debug(f"[QDRANT] Top result score: {output[0]['score']:.4f}")

        return output


    def _build_filter(self, metadata_filter: Dict[str, Any]) -> Filter:
        """
        Convert a metadata filter dict (Elasticsearch-style operators) to a
        Qdrant ``Filter`` object.

        Supported operators:
          - Simple equality:  ``{"field": value}``
          - Range:            ``{"field": {"$gte": v, "$lte": v, "$gt": v, "$lt": v}}``
          - In-list:          ``{"field": {"$in": [v1, v2, ...]}}``
        """
        must_conditions = []

        for key, value in metadata_filter.items():
            if isinstance(value, dict):
                # Range queries
                if any(op in value for op in ("$gte", "$lte", "$gt", "$lt")):
                    range_kwargs: Dict[str, Any] = {}
                    if "$gte" in value:
                        range_kwargs["gte"] = value["$gte"]
                    if "$lte" in value:
                        range_kwargs["lte"] = value["$lte"]
                    if "$gt" in value:
                        range_kwargs["gt"] = value["$gt"]
                    if "$lt" in value:
                        range_kwargs["lt"] = value["$lt"]
                    must_conditions.append(
                        FieldCondition(key=key, range=Range(**range_kwargs))
                    )
                # $in list queries
                elif "$in" in value:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value["$in"]))
                    )
            else:
                # Simple equality
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=must_conditions)

    @staticmethod
    def _extract_text(document: Dict[str, Any]) -> str:
        """
        Concatenate all searchable text fields from a patent document.
        Mirrors the search_fields used in the old Elasticsearch query:
        title^2, abstract, text, claims, patent_id.
        """
        parts: List[str] = []
        # Weight title twice by repeating it
        title = document.get("title", "")
        if title:
            parts.extend([title, title])
        for field in ("abstract", "text", "claims", "patent_id"):
            val = document.get(field)
            if val:
                parts.append(str(val))
        return " ".join(parts)
