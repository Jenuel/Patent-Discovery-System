"""
Sample script to populate MongoDB with patent chunk data.

This script demonstrates how to insert patent chunks into MongoDB
that correspond to the vectors stored in Pinecone.

Usage:
    python populate_mongodb.py --input patents.jsonl
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from app.services.storage.mongodb import MongoDBStore


async def load_chunks_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load patent chunks from a JSONL file.
    
    Expected format per line:
    {
        "id": "US20210123456A1::abstract::0000",
        "text": "Patent text content...",
        "patent_id": "US20210123456A1",
        "section": "abstract",
        "chunk_id": 0,
        "cpc": ["G06N3/08"],
        "filing_year": 2019,
        "char_start": 0,
        "char_end": 76,
        "title": "Patent Title",
        "claim_no": null
    }
    """
    chunks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Ensure _id field is set
                if 'id' in data:
                    data['_id'] = data.pop('id')
                elif '_id' not in data:
                    # Generate ID from components
                    patent_id = data.get('patent_id')
                    section = data.get('section')
                    chunk_id = data.get('chunk_id', 0)
                    data['_id'] = f"{patent_id}::{section}::{chunk_id:04d}"
                
                chunks.append(data)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return chunks


async def populate_mongodb(chunks: List[Dict[str, Any]], batch_size: int = 1000):
    """
    Populate MongoDB with patent chunks in batches.
    
    Args:
        chunks: List of chunk documents to insert
        batch_size: Number of documents to insert per batch
    """
    mongo_store = MongoDBStore.from_env()
    
    total_chunks = len(chunks)
    print(f"Inserting {total_chunks} chunks into MongoDB...")
    
    # Insert in batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        
        try:
            await mongo_store.insert_chunks(batch)
            print(f"Inserted batch {i // batch_size + 1}: {len(batch)} chunks")
        except Exception as e:
            print(f"Error inserting batch {i // batch_size + 1}: {e}")
            # Try inserting one by one to identify problematic documents
            for chunk in batch:
                try:
                    await mongo_store.insert_chunk(chunk['_id'], chunk)
                except Exception as chunk_error:
                    print(f"Failed to insert chunk {chunk.get('_id')}: {chunk_error}")
    
    await mongo_store.close()
    print("MongoDB population complete!")


async def verify_insertion(sample_ids: List[str]):
    """
    Verify that chunks were inserted correctly by fetching a sample.
    
    Args:
        sample_ids: List of chunk IDs to verify
    """
    mongo_store = MongoDBStore.from_env()
    
    print(f"\nVerifying {len(sample_ids)} sample chunks...")
    chunks_map = await mongo_store.get_chunks_by_ids(sample_ids)
    
    for chunk_id in sample_ids:
        chunk = chunks_map.get(chunk_id)
        if chunk:
            print(f"✓ Found: {chunk_id}")
            print(f"  Patent: {chunk.get('patent_id')}")
            print(f"  Section: {chunk.get('section')}")
            print(f"  Text length: {len(chunk.get('text', ''))} chars")
        else:
            print(f"✗ Missing: {chunk_id}")
    
    await mongo_store.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with patent chunk data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL file containing patent chunks"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents to insert per batch (default: 1000)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify insertion by sampling some chunks"
    )
    
    args = parser.parse_args()
    
    # Load chunks from file
    print(f"Loading chunks from {args.input}...")
    chunks = await load_chunks_from_jsonl(args.input)
    
    if not chunks:
        print("No chunks loaded. Exiting.")
        return
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Populate MongoDB
    await populate_mongodb(chunks, batch_size=args.batch_size)
    
    # Verify if requested
    if args.verify and chunks:
        # Sample first 5 chunks for verification
        sample_ids = [chunk['_id'] for chunk in chunks[:5]]
        await verify_insertion(sample_ids)


if __name__ == "__main__":
    asyncio.run(main())
