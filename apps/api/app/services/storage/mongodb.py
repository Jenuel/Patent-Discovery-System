from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from app.core.logging import get_logger

log = get_logger(__name__)


class MongoDBStore:
    """
    MongoDB client for storing and retrieving patent text data.
    
    This service stores the raw text content of patent chunks that are
    embedded and indexed in Pinecone. The chunk IDs from Pinecone are used
    to retrieve the corresponding text from MongoDB.
    """

    def __init__(
        self,
        connection_string: str,
        database_name: str = "patent_discovery",
        collection_name: str = "patent_chunks",
    ):
        """
        Initialize MongoDB client.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database
            collection_name: Name of the collection storing patent chunks
        """
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(connection_string)
        self.db: AsyncIOMotorDatabase = self.client[database_name]
        self.collection: AsyncIOMotorCollection = self.db[collection_name]

    @classmethod
    def from_env(cls) -> "MongoDBStore":
        """
        Create MongoDB store from environment variables.
        
        Expected environment variables:
        - MONGODB_URI: MongoDB connection string
        - MONGODB_DATABASE: Database name (default: patent_discovery)
        - MONGODB_COLLECTION: Collection name (default: patent_chunks)
        """
        connection_string = os.getenv("MONGODB_URI")
        if not connection_string:
            raise ValueError("MONGODB_URI environment variable is required")
        
        database_name = os.getenv("MONGODB_DATABASE", "patent_discovery")
        collection_name = os.getenv("MONGODB_COLLECTION", "patent_chunks")
        
        return cls(
            connection_string=connection_string,
            database_name=database_name,
            collection_name=collection_name,
        )

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single patent chunk by its ID.
        
        Args:
            chunk_id: The chunk ID (e.g., "US20210123456A1::abstract::0000")
            
        Returns:
            Document containing the chunk data, or None if not found
        """
        log.debug(f"[MONGODB] Fetching chunk by ID: {chunk_id}")
        result = await self.collection.find_one({"id": chunk_id})
        if result:
            log.debug(f"[MONGODB] Chunk found: {chunk_id}")
        else:
            log.warning(f"[MONGODB] Chunk not found: {chunk_id}")
        return result

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve multiple patent chunks by their IDs in a single query.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            Dictionary mapping chunk_id to document data
        """
        if not chunk_ids:
            log.debug("[MONGODB] No chunk IDs provided")
            return {}
        
        log.info(f"[MONGODB] Fetching {len(chunk_ids)} chunks in batch query")
        cursor = self.collection.find({"id": {"$in": chunk_ids}})
        
        # Build a mapping of chunk_id -> document
        chunks_map: Dict[str, Dict[str, Any]] = {}
        async for doc in cursor:
            chunk_id = doc.get("id")
            if chunk_id:
                chunks_map[chunk_id] = doc
        
        log.info(f"[MONGODB] Retrieved {len(chunks_map)}/{len(chunk_ids)} chunks")
        if len(chunks_map) < len(chunk_ids):
            missing_count = len(chunk_ids) - len(chunks_map)
            log.warning(f"[MONGODB] {missing_count} chunks not found in database")
        
        return chunks_map

    async def insert_chunk(self, chunk_id: str, data: Dict[str, Any]) -> None:
        """
        Insert a single patent chunk.
        
        Args:
            chunk_id: The chunk ID
            data: Document data to store
        """
        log.debug(f"[MONGODB] Inserting chunk: {chunk_id}")
        document = {"id": chunk_id, **data}
        await self.collection.insert_one(document)
        log.debug(f"[MONGODB] Chunk inserted: {chunk_id}")

    async def insert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Insert multiple patent chunks in bulk.
        
        Args:
            chunks: List of documents, each must have an "id" field
        """
        if not chunks:
            log.debug("[MONGODB] No chunks to insert")
            return
        
        log.info(f"[MONGODB] Bulk inserting {len(chunks)} chunks")
        await self.collection.insert_many(chunks, ordered=False)
        log.info(f"[MONGODB] Bulk insert complete: {len(chunks)} chunks")

    async def close(self) -> None:
        """Close the MongoDB connection."""
        self.client.close()
