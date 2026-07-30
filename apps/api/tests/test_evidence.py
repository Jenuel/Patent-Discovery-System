"""Tests for RAGOrchestrator._to_evidence_items — the claim-evidence path.

Covers the id contract between Qdrant point ids and MongoDB chunk _ids, and
the field-name drift between the chunk documents and the EvidenceItem schema.

Run from apps/api: python -m unittest discover tests
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List

from app.services.rag.orchestrator import RAGOrchestrator
from app.services.retrieval.fusion import ScoredMatch


class FakeMongo:
    """Keys chunks by _id, exactly as MongoDBStore.get_chunks_by_ids now does."""

    def __init__(self, docs: List[Dict[str, Any]]):
        self._by_id = {str(d["_id"]): d for d in docs}
        self.requested: List[str] = []

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        self.requested = list(chunk_ids)
        return {cid: self._by_id[cid] for cid in chunk_ids if cid in self._by_id}


def _orchestrator(mongo: FakeMongo) -> RAGOrchestrator:
    return RAGOrchestrator(
        embedder=object(),
        qdrant_store=object(),
        mongodb_store=mongo,
        llm=object(),
    )


def _chunk(**overrides: Any) -> Dict[str, Any]:
    """A chunk in the shape populate_mongodb.py actually writes: flat, _id-keyed."""
    doc = {
        "_id": "US123::claim::0001",
        "text": "A battery cathode comprising lithium.",
        "patent_id": "US123",
        "section": "claim",
        "cpc": ["G06N3/08"],
        "filing_year": 2019,
        "title": "Battery Cathode",
        "claim_no": 1,
    }
    doc.update(overrides)
    return doc


class EvidenceFromFlatChunksTests(unittest.IsolatedAsyncioTestCase):
    async def test_flat_chunk_populates_every_field(self):
        mongo = FakeMongo([_chunk()])
        orch = _orchestrator(mongo)

        match = ScoredMatch(id="US123::claim::0001", score=0.87, metadata={})
        items = await orch._to_evidence_items([match], source="hybrid")

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.chunk_id, "US123::claim::0001")
        self.assertEqual(item.patent_id, "US123")
        self.assertEqual(item.title, "Battery Cathode")
        self.assertEqual(item.level, "claim")
        self.assertEqual(item.text, "A battery cathode comprising lithium.")
        self.assertEqual(item.score, 0.87)

    async def test_claim_no_is_read_from_the_key_chunks_actually_use(self):
        """Chunks store `claim_no`; the schema field is `claim_no` too.

        The reader previously looked for `claim_number` only, so this was
        always None.
        """
        mongo = FakeMongo([_chunk(claim_no=7)])
        orch = _orchestrator(mongo)

        items = await orch._to_evidence_items(
            [ScoredMatch(id="US123::claim::0001", score=0.5, metadata={})],
            source="hybrid",
        )
        self.assertEqual(items[0].claim_no, 7)

    async def test_claim_number_alias_still_works(self):
        doc = _chunk()
        del doc["claim_no"]
        doc["claim_number"] = 3
        orch = _orchestrator(FakeMongo([doc]))

        items = await orch._to_evidence_items(
            [ScoredMatch(id="US123::claim::0001", score=0.5, metadata={})],
            source="hybrid",
        )
        self.assertEqual(items[0].claim_no, 3)

    async def test_nested_metadata_documents_still_work(self):
        doc = {
            "_id": "US999::claim::0002",
            "metadata": {
                "patent_id": "US999",
                "title": "Nested Title",
                "text": "Nested claim text.",
                "claim_no": 2,
                "section": "claim",
            },
        }
        orch = _orchestrator(FakeMongo([doc]))

        items = await orch._to_evidence_items(
            [ScoredMatch(id="US999::claim::0002", score=0.5, metadata={})],
            source="hybrid",
        )
        self.assertEqual(items[0].patent_id, "US999")
        self.assertEqual(items[0].text, "Nested claim text.")
        self.assertEqual(items[0].claim_no, 2)


class MongoMissFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_chunk_falls_back_to_the_qdrant_payload(self):
        """A MongoDB miss must not silently produce empty evidence.

        Empty text validates fine against the schema, so the pipeline would
        report success and hand the LLM nothing but the query.
        """
        orch = _orchestrator(FakeMongo([]))

        match = ScoredMatch(
            id="US123::claim::0001",
            score=0.5,
            metadata={
                "patent_id": "US123",
                "title": "Battery Cathode",
                "text": "A battery cathode comprising lithium.",
                "claim_no": 1,
                "section": "claim",
            },
        )
        items = await orch._to_evidence_items([match], source="hybrid")

        self.assertEqual(items[0].text, "A battery cathode comprising lithium.")
        self.assertEqual(items[0].patent_id, "US123")
        self.assertEqual(items[0].claim_no, 1)

    async def test_total_miss_yields_empty_text_not_a_crash(self):
        orch = _orchestrator(FakeMongo([]))
        items = await orch._to_evidence_items(
            [ScoredMatch(id="nope", score=0.1, metadata={})],
            source="hybrid",
        )
        self.assertEqual(items[0].text, "")
        self.assertEqual(items[0].patent_id, "")

    async def test_lookup_uses_the_match_ids_verbatim(self):
        mongo = FakeMongo([_chunk()])
        orch = _orchestrator(mongo)

        await orch._to_evidence_items(
            [ScoredMatch(id="US123::claim::0001", score=0.5, metadata={})],
            source="hybrid",
        )
        self.assertEqual(mongo.requested, ["US123::claim::0001"])

    async def test_no_matches_short_circuits(self):
        mongo = FakeMongo([_chunk()])
        orch = _orchestrator(mongo)
        self.assertEqual(await orch._to_evidence_items([], source="hybrid"), [])
        self.assertEqual(mongo.requested, [])


if __name__ == "__main__":
    unittest.main()
