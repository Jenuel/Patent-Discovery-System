"""
index_qdrant.py
===============
Index patents and claim-level chunks into the Qdrant hybrid collections.

Collections created / populated:
  patents_hybrid  — patent-level dense (OpenAI) + BM25 sparse vectors
  claims_hybrid   — claim-level dense (OpenAI) vectors only

Usage
-----
# Index patents from a JSONL file:
    python scripts/index_qdrant.py patents --input data/patents.jsonl

# Index claim chunks from MongoDB:
    python scripts/index_qdrant.py claims

# Index both in one shot:
    python scripts/index_qdrant.py all --input data/patents.jsonl

# Recreate collections before indexing (full re-index):
    python scripts/index_qdrant.py all --input data/patents.jsonl --recreate

# Index only claim-section chunks (the script prints the section distribution
# it found, so run it once without the flag to see what's in MongoDB):
    python scripts/index_qdrant.py claims --sections claim

Environment variables required
-------------------------------
    QDRANT_URL
    QDRANT_API_KEY
    QDRANT_PATENT_COLLECTION_NAME  (default: patents_hybrid)
    QDRANT_CLAIM_COLLECTION_NAME   (default: claims_hybrid)
    QDRANT_DENSE_VECTOR_SIZE       (default: 1536)
    OPENAI_API_KEY
    MONGODB_URI                    (required for claim indexing)
    MONGODB_DATABASE               (default: patent_discovery)
    MONGODB_COLLECTION             (default: patent_chunks)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.indexing.embed import embed_texts
from app.services.indexing.qdrant import QdrantHybridStore
from app.services.storage.mongodb import MongoDBStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping line {line_no}: {exc}", flush=True)
    return records


def _normalise_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure each document has an ``_id`` field."""
    if "_id" not in doc:
        if "id" in doc:
            doc["_id"] = doc.pop("id")
        elif "patent_id" in doc:
            doc["_id"] = doc["patent_id"]
        else:
            raise ValueError(f"Document has no id / patent_id field: {list(doc.keys())}")
    return doc


_extract_patent_text_dense = QdrantHybridStore._extract_patent_text_dense
_extract_claim_text = QdrantHybridStore._extract_claim_text


def _partition_by_text(
    docs: List[Dict[str, Any]],
    extract: Any,
    label: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Split documents into those with usable text and those without.

    ``embed_batch`` raises on the *whole* batch if any single input is empty
    after cleaning, so one text-less document would abort the entire run.
    Drop them here instead, loudly.
    """
    kept: List[Dict[str, Any]] = []
    texts: List[str] = []
    skipped: List[str] = []

    for doc in docs:
        text = (extract(doc) or "").strip()
        if not text:
            skipped.append(str(doc.get("_id", "<no id>")))
            continue
        kept.append(doc)
        texts.append(text)

    if skipped:
        preview = ", ".join(skipped[:5])
        more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
        print(
            f"  [WARN] Skipping {len(skipped)} {label} with no text: {preview}{more}",
            flush=True,
        )

    return kept, texts


def _section_of(doc: Dict[str, Any]) -> str:
    """Return a chunk's section/level label, however it happens to be spelled."""
    return str(doc.get("section") or doc.get("level") or "<unset>").lower()


# ---------------------------------------------------------------------------
# Patent indexing
# ---------------------------------------------------------------------------

async def index_patents(
    store: QdrantHybridStore,
    input_path: str,
    batch_size: int,
) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"PATENT INDEXING  →  {store.cfg.patent_collection_name}", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"Loading patents from '{input_path}'…", flush=True)
    docs = [_normalise_id(d) for d in _load_jsonl(input_path)]
    print(f"Loaded {len(docs)} patent documents", flush=True)

    if not docs:
        print("No documents to index. Exiting patent step.", flush=True)
        return

    # Compute dense embeddings in batches. The BM25 arm uses a *different*
    # text (title-weighted) and is computed inside bulk_index_patents.
    docs, texts = _partition_by_text(docs, _extract_patent_text_dense, "patents")
    if not docs:
        print("No patents with usable text. Exiting patent step.", flush=True)
        return

    print(f"Computing dense embeddings for {len(docs)} patents…", flush=True)
    import anyio

    dense_vectors = await anyio.to_thread.run_sync(lambda: embed_texts(texts))
    print(f"Embeddings computed (dim={len(dense_vectors[0])})", flush=True)

    # Bulk index (also computes BM25 sparse internally)
    print(f"Upserting to Qdrant (batch_size={batch_size})…", flush=True)
    await store.bulk_index_patents(docs, dense_vectors=dense_vectors)
    print(f"✓ Patent indexing complete: {len(docs)} documents", flush=True)


# ---------------------------------------------------------------------------
# Claim indexing
# ---------------------------------------------------------------------------

async def index_claims(
    store: QdrantHybridStore,
    batch_size: int,
    sections: Optional[List[str]] = None,
) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"CLAIM INDEXING  →  {store.cfg.claim_collection_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load chunks from MongoDB
    mongodb = MongoDBStore.from_env()
    try:
        print("Loading chunks from MongoDB…", flush=True)
        docs: List[Dict[str, Any]] = []
        no_id = 0
        async for raw in mongodb.collection.find({}):
            doc_id = raw.get("_id") or raw.get("id")
            if not doc_id:
                no_id += 1
                continue
            raw["_id"] = str(doc_id)
            docs.append(raw)
    finally:
        await mongodb.close()

    if no_id:
        print(f"  [WARN] Skipped {no_id} chunks with no id", flush=True)
    print(f"Loaded {len(docs)} chunk documents from MongoDB", flush=True)

    if not docs:
        print("No chunks found in MongoDB. Exiting claim step.", flush=True)
        return

    counts: Dict[str, int] = {}
    for doc in docs:
        section = _section_of(doc)
        counts[section] = counts.get(section, 0) + 1
    print("Chunk sections found: " + ", ".join(
        f"{name}={n}" for name, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ), flush=True)

    if sections:
        wanted = {s.strip().lower() for s in sections}
        docs = [d for d in docs if _section_of(d) in wanted]
        print(f"Filtered to sections {sorted(wanted)}: {len(docs)} chunks", flush=True)
        if not docs:
            print(
                "  [ERROR] No chunks matched --sections. Check the section names "
                "listed above and re-run.",
                flush=True,
            )
            return
    elif len(counts) > 1:
        print(
            "  [NOTE] Multiple sections present and no --sections filter given, so "
            "ALL of them will be indexed into the claim collection. Pass e.g. "
            "--sections claim to restrict it.",
            flush=True,
        )

    # Compute dense embeddings
    docs, texts = _partition_by_text(docs, _extract_claim_text, "chunks")
    if not docs:
        print("No chunks with usable text. Exiting claim step.", flush=True)
        return

    print(f"Computing dense embeddings for {len(docs)} chunks…", flush=True)
    import anyio

    dense_vectors = await anyio.to_thread.run_sync(lambda: embed_texts(texts))
    print(f"Embeddings computed (dim={len(dense_vectors[0])})", flush=True)

    # Bulk index
    print(f"Upserting to Qdrant (batch_size={batch_size})…", flush=True)
    await store.bulk_index_claims(docs, dense_vectors=dense_vectors)
    print(f"✓ Claim indexing complete: {len(docs)} documents", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index patents and/or claims into Qdrant hybrid collections."
    )
    parser.add_argument(
        "mode",
        choices=["patents", "claims", "all"],
        help="What to index: 'patents', 'claims', or 'all'",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to JSONL file containing patent documents (required for 'patents' and 'all' modes)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Upsert batch size (default: 100)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete and recreate Qdrant collections before indexing. "
            "Required when changing the collection schema (e.g. vector size)."
        ),
    )
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help=(
            "Comma-separated chunk sections to index into the claim collection "
            "(e.g. 'claim'). Default: index every section found in MongoDB. "
            "The script prints the section distribution before filtering."
        ),
    )

    args = parser.parse_args()

    # Validate
    if args.mode in ("patents", "all") and not args.input:
        parser.error("--input is required for 'patents' and 'all' modes")

    # Build store
    store = QdrantHybridStore.from_env()

    # Recreate if requested
    if args.recreate:
        print("\n[RECREATE] Deleting existing collections…", flush=True)
        await store.delete_collections()

    # Ensure collections exist
    print("\nEnsuring Qdrant collections exist…", flush=True)
    await store.create_collections()

    # Index
    if args.mode in ("patents", "all"):
        await index_patents(store, args.input, args.batch_size)

    if args.mode in ("claims", "all"):
        sections = args.sections.split(",") if args.sections else None
        await index_claims(store, args.batch_size, sections=sections)

    await store.close()
    print("\n✓ All indexing complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
