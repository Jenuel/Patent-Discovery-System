"""Unit tests for QdrantHybridStore's query-building logic.

These exercise pure request construction — no Qdrant server and no fastembed
model are needed (the BM25 encoder is imported lazily).

Run from apps/api: python -m unittest discover tests
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List

from qdrant_client.models import FusionQuery, MatchAny, MatchValue, PayloadSchemaType

from app.services.indexing.qdrant import QdrantHybridStore
from app.services.indexing.schemas import QdrantConfig


def _store(client: Any = None) -> QdrantHybridStore:
    """Build a store with an injected client so no test touches the network."""
    return QdrantHybridStore(
        QdrantConfig(url="http://localhost:6333", api_key=""),
        client=client or FakeClient(),
    )


class _HttpError(Exception):
    """Stands in for qdrant_client's UnexpectedResponse."""

    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


class FakeClient:
    """Records index/delete calls so tests can assert on them."""

    def __init__(
        self,
        payload_schema: Dict[str, Dict[str, Any]] | None = None,
        get_error: Exception | None = None,
        index_error: Exception | None = None,
        delete_error: Exception | None = None,
    ):
        self._payload_schema = payload_schema or {}
        self._get_error = get_error
        self._index_error = index_error
        self._delete_error = delete_error
        self.created: List[tuple] = []
        self.deleted: List[str] = []

    async def get_collection(self, collection_name: str):
        if self._get_error:
            raise self._get_error

        class Info:
            payload_schema = self._payload_schema.get(collection_name, {})

        return Info()

    async def create_payload_index(self, collection_name, field_name, field_schema):
        if self._index_error:
            raise self._index_error
        self.created.append((collection_name, field_name, field_schema))

    async def delete_collection(self, collection_name: str):
        if self._delete_error:
            raise self._delete_error
        self.deleted.append(collection_name)


class BuildFilterTests(unittest.TestCase):
    def test_simple_equality(self):
        f = _store()._build_filter({"patent_id": "US1"})
        self.assertEqual(f.must[0].key, "patent_id")
        self.assertEqual(f.must[0].match, MatchValue(value="US1"))

    def test_explicit_eq_operator(self):
        f = _store()._build_filter({"patent_id": {"$eq": "US1"}})
        self.assertEqual(f.must[0].match, MatchValue(value="US1"))

    def test_in_operator(self):
        f = _store()._build_filter({"patent_id": {"$in": ["US1", "US2"]}})
        self.assertEqual(f.must[0].match, MatchAny(any=["US1", "US2"]))

    def test_range_operators(self):
        f = _store()._build_filter({"year": {"$gte": 2010, "$lt": 2020}})
        self.assertEqual(f.must[0].range.gte, 2010)
        self.assertEqual(f.must[0].range.lt, 2020)
        self.assertIsNone(f.must[0].range.lte)

    def test_unsupported_operator_raises_instead_of_being_dropped(self):
        # A dropped condition widens the search rather than narrowing it, which
        # would silently defeat the Stage-2 patent_id allowlist.
        with self.assertRaises(ValueError):
            _store()._build_filter({"patent_id": {"$ne": "US1"}})

    def test_unknown_operator_mixed_with_range_raises(self):
        with self.assertRaises(ValueError):
            _store()._build_filter({"year": {"$gte": 2010, "$nin": [2015]}})


class FusionQueryTests(unittest.TestCase):
    def test_bare_fusion_enum_would_resolve_to_a_point_id_lookup(self):
        """Guards the reason search_hybrid must send FusionQuery, not Fusion.

        Fusion subclasses str, so qdrant-client's query resolver matches it
        against PointId and issues a nearest-by-id lookup instead of fusing.
        """
        from qdrant_client.models import Fusion
        from qdrant_client.qdrant_fastembed import QdrantFastembedMixin

        resolved = QdrantFastembedMixin._resolve_query(Fusion.RRF)
        self.assertNotIsInstance(resolved, FusionQuery)

        wrapped = QdrantFastembedMixin._resolve_query(FusionQuery(fusion=Fusion.RRF))
        self.assertIsInstance(wrapped, FusionQuery)


class ConfigValidationTests(unittest.TestCase):
    def test_missing_url_raises(self):
        with self.assertRaises(ValueError):
            QdrantHybridStore(QdrantConfig(url="", api_key="key"))


class NormaliseResultsTests(unittest.TestCase):
    def test_doc_id_is_lifted_out_of_the_payload(self):
        class FakePoint:
            id = 123
            score = 0.42
            payload = {"_doc_id": "chunk-abc", "patent_id": "US1"}

        out = _store()._normalise_results([FakePoint()])
        self.assertEqual(out[0]["id"], "chunk-abc")
        self.assertEqual(out[0]["score"], 0.42)
        self.assertNotIn("_doc_id", out[0]["metadata"])
        self.assertEqual(out[0]["metadata"]["patent_id"], "US1")

    def test_falls_back_to_point_id_when_doc_id_absent(self):
        class FakePoint:
            id = 123
            score = 0.1
            payload = {"patent_id": "US1"}

        out = _store()._normalise_results([FakePoint()])
        self.assertEqual(out[0]["id"], "123")


class _FakeEmbedding:
    """Mimics a fastembed SparseEmbedding (numpy arrays with .tolist())."""

    class _Arr:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return list(self._data)

    def __init__(self, indices, values):
        self.indices = self._Arr(indices)
        self.values = self._Arr(values)


class SparseVectorConversionTests(unittest.TestCase):
    def test_normal_embedding_converts(self):
        from app.services.indexing.qdrant import _to_sparse_vector

        vec = _to_sparse_vector(_FakeEmbedding([3, 9], [0.5, 0.25]))
        self.assertEqual(vec.indices, [3, 9])
        self.assertEqual(vec.values, [0.5, 0.25])

    def test_termless_embedding_returns_none_not_a_dummy_token(self):
        """A placeholder token would corrupt collection-wide IDF statistics.

        The collection uses Modifier.IDF, so Qdrant derives inverse document
        frequency from document counts. Seeding every term-less document with
        the same dummy token inflates that token's document frequency and
        perturbs the scores of every other query.
        """
        from app.services.indexing.qdrant import _to_sparse_vector

        self.assertIsNone(_to_sparse_vector(_FakeEmbedding([], [])))


class PatentTextExtractionTests(unittest.TestCase):
    DOC = {
        "title": "Battery Cathode",
        "abstract": "A cathode comprising lithium.",
        "patent_id": "US20210123456A1",
    }

    def test_sparse_text_repeats_the_title_for_bm25_weighting(self):
        text = QdrantHybridStore._extract_patent_text_sparse(self.DOC)
        self.assertEqual(text.count("Battery Cathode"), 2)

    def test_sparse_text_includes_the_patent_id_for_lexical_lookup(self):
        text = QdrantHybridStore._extract_patent_text_sparse(self.DOC)
        self.assertIn("US20210123456A1", text)

    def test_dense_text_states_the_title_once(self):
        """Repeating the title skews the embedding toward it.

        Token repetition is a BM25 term-frequency weighting device; it has no
        equivalent meaning for a dense embedding.
        """
        text = QdrantHybridStore._extract_patent_text_dense(self.DOC)
        self.assertEqual(text.count("Battery Cathode"), 1)

    def test_dense_text_excludes_the_patent_id(self):
        """An alphanumeric publication number carries no semantic signal."""
        text = QdrantHybridStore._extract_patent_text_dense(self.DOC)
        self.assertNotIn("US20210123456A1", text)

    def test_the_two_builders_actually_differ(self):
        self.assertNotEqual(
            QdrantHybridStore._extract_patent_text_sparse(self.DOC),
            QdrantHybridStore._extract_patent_text_dense(self.DOC),
        )

    def test_dense_text_keeps_the_substantive_fields(self):
        text = QdrantHybridStore._extract_patent_text_dense(self.DOC)
        self.assertIn("Battery Cathode", text)
        self.assertIn("A cathode comprising lithium.", text)


class NotFoundDetectionTests(unittest.TestCase):
    def test_404_is_absence(self):
        self.assertTrue(QdrantHybridStore._is_not_found(_HttpError(404)))

    def test_401_is_not_absence_even_if_the_body_says_not_found(self):
        """A status code, when present, is trusted exclusively.

        An auth failure whose body happens to contain "not found" must not be
        read as "the collection does not exist".
        """
        self.assertFalse(
            QdrantHybridStore._is_not_found(_HttpError(401, "api key not found"))
        )

    def test_falls_back_to_the_message_when_there_is_no_status_code(self):
        self.assertTrue(QdrantHybridStore._is_not_found(Exception("Collection not found")))
        self.assertFalse(QdrantHybridStore._is_not_found(Exception("connection refused")))


class PayloadIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_indexes_are_per_collection(self):
        """Patent field names must not be indexed on the claim collection.

        The two payloads have different shapes; the claim collection filters on
        patent_id alone (Stage-1 allowlist), so indexing `cpc`/`year` there
        would index nothing.
        """
        client = FakeClient()
        store = _store(client)

        await store._create_payload_indexes()

        patents = store.cfg.patent_collection_name
        claims = store.cfg.claim_collection_name

        claim_fields = {f for (c, f, _) in client.created if c == claims}
        patent_fields = {f for (c, f, _) in client.created if c == patents}

        self.assertEqual(claim_fields, {"patent_id"})
        self.assertIn("patent_id", patent_fields)
        self.assertIn("year", patent_fields)
        self.assertTrue({"cpc", "cpc_prefix"} <= patent_fields)

    async def test_existing_indexes_are_not_recreated(self):
        client = FakeClient(
            payload_schema={
                "patents_hybrid": {"patent_id": object()},
                "claims_hybrid": {"patent_id": object()},
            }
        )
        store = _store(client)

        await store._create_payload_indexes()

        created_fields = [(c, f) for (c, f, _) in client.created]
        self.assertNotIn((store.cfg.claim_collection_name, "patent_id"), created_fields)
        self.assertNotIn((store.cfg.patent_collection_name, "patent_id"), created_fields)
        # The remaining patent fields are still created.
        self.assertIn((store.cfg.patent_collection_name, "year"), created_fields)

    async def test_index_creation_failure_propagates(self):
        """A failed payload index must not be downgraded to a debug log.

        Missing indexes are invisible until Stage 2 gets slow.
        """
        store = _store(FakeClient(index_error=_HttpError(403, "forbidden")))

        with self.assertRaises(_HttpError):
            await store._create_payload_indexes()

    async def test_schema_types_are_correct(self):
        client = FakeClient()
        store = _store(client)

        await store._create_payload_indexes()

        schemas = {(c, f): s for (c, f, s) in client.created}
        self.assertEqual(
            schemas[(store.cfg.patent_collection_name, "patent_id")],
            PayloadSchemaType.KEYWORD,
        )
        self.assertEqual(
            schemas[(store.cfg.patent_collection_name, "year")],
            PayloadSchemaType.INTEGER,
        )


class DeleteCollectionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_deletes_both_collections(self):
        client = FakeClient()
        store = _store(client)

        await store.delete_collections()

        self.assertEqual(
            client.deleted,
            [store.cfg.patent_collection_name, store.cfg.claim_collection_name],
        )

    async def test_missing_collection_is_tolerated(self):
        store = _store(FakeClient(delete_error=_HttpError(404)))
        await store.delete_collections()  # must not raise

    async def test_real_failure_raises_instead_of_silently_skipping(self):
        """--recreate must not degrade into an incremental update.

        A swallowed delete leaves the collection in place, so create_collections
        skips creation and indexing upserts into the stale collection — points
        from a previous corpus survive a supposed full re-index.
        """
        store = _store(FakeClient(delete_error=_HttpError(401, "unauthorized")))

        with self.assertRaises(_HttpError):
            await store.delete_collections()


if __name__ == "__main__":
    unittest.main()
