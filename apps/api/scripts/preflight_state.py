#!/usr/bin/env python3
"""
Read-only pre-flight: report the current state of Qdrant and MongoDB.

Creates, modifies and deletes NOTHING. Run before any indexing session --
index_qdrant.py claims reads the ENTIRE Mongo collection via find({}), so
pre-existing documents get embedded (billed) and indexed (RAM) whether you
meant them to or not.

    python scripts/preflight_state.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

patent_coll = os.getenv("QDRANT_PATENT_COLLECTION_NAME", "patents_hybrid")
claim_coll = os.getenv("QDRANT_CLAIM_COLLECTION_NAME", "claims_hybrid")
mongo_db = os.getenv("MONGODB_DATABASE", "patent_database")
mongo_coll = os.getenv("MONGODB_COLLECTION", "patent_chunks")

print("=" * 70)
print("CONFIGURED TARGETS")
print("=" * 70)
print(f"  QDRANT_PATENT_COLLECTION_NAME = {patent_coll}")
print(f"  QDRANT_CLAIM_COLLECTION_NAME  = {claim_coll}")
print(f"  QDRANT_DENSE_VECTOR_SIZE      = {os.getenv('QDRANT_DENSE_VECTOR_SIZE', '1536')}")
print(f"  MONGODB_DATABASE              = {mongo_db}")
print(f"  MONGODB_COLLECTION            = {mongo_coll}")

print("\n" + "=" * 70)
print("QDRANT")
print("=" * 70)
try:
    from qdrant_client import QdrantClient

    qc = QdrantClient(url=os.getenv("QDRANT_URL"),
                      api_key=os.getenv("QDRANT_API_KEY"), timeout=30)
    existing = [c.name for c in qc.get_collections().collections]
    print(f"  Collections present ({len(existing)}): {existing or '(none)'}")

    for name in existing:
        info = qc.get_collection(name)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors
        vdesc = (", ".join(f"{k}(size={v.size},{v.distance})" for k, v in vectors.items())
                 if hasattr(vectors, "items")
                 else f"unnamed(size={getattr(vectors, 'size', '?')})")
        tag = ("  <-- configured PATENT target" if name == patent_coll else
               "  <-- configured CLAIM target" if name == claim_coll else "")
        print(f"\n  [{name}]{tag}")
        print(f"      points        : {info.points_count:,}")
        print(f"      dense vectors : {vdesc}")
        print(f"      sparse vectors: {', '.join(sparse.keys()) if sparse else '(none)'}")
        print(f"      payload idx   : {sorted((info.payload_schema or {}).keys()) or '(none)'}")

    for label, name in (("patent", patent_coll), ("claim", claim_coll)):
        if name not in existing:
            print(f"\n  NOTE: configured {label} collection '{name}' does NOT exist yet.")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {str(e)[:400]}")

print("\n" + "=" * 70)
print("MONGODB")
print("=" * 70)
try:
    from pymongo import MongoClient

    mc = MongoClient(os.getenv("MONGODB_URI"), serverSelectionTimeoutMS=15000)
    db = mc[mongo_db]
    colls = db.list_collection_names()
    print(f"  Database '{mongo_db}' collections ({len(colls)}): {colls or '(none)'}")

    for name in colls:
        n = db[name].count_documents({})
        tag = "  <-- configured target" if name == mongo_coll else ""
        print(f"\n  [{name}]{tag}")
        print(f"      documents : {n:,}")
        if n:
            try:
                st = db.command("collstats", name)
                print(f"      dataSize  : {st.get('size', 0) / 1e6:.1f} MB")
                print(f"      storage   : {st.get('storageSize', 0) / 1e6:.1f} MB "
                      f"(+ indexes {st.get('totalIndexSize', 0) / 1e6:.1f} MB)")
            except Exception:
                pass
            print(f"      sections  : {db[name].distinct('section') or '(none)'}")
            print(f"      distinct patent_id: {len(db[name].distinct('patent_id')):,}")
            sample = db[name].find_one({})
            if sample:
                print(f"      keys      : {sorted(sample.keys())}")
                print(f"      sample _id: {sample.get('_id')!r}")

    if mongo_coll not in colls:
        print(f"\n  NOTE: configured collection '{mongo_coll}' does NOT exist yet.")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {str(e)[:400]}")

print("\n" + "=" * 70)
print("Read-only. Nothing was created, modified, or deleted.")
print("=" * 70)
