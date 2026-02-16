import os
from typing import Any, Dict, List, Optional
from elasticsearch import AsyncElasticsearch

from app.core.logging import get_logger
from .schemas import ElasticsearchConfig

log = get_logger(__name__)


class ElasticsearchStore:
    """
    Elasticsearch wrapper for BM25 sparse retrieval.
    Used for patent-level lexical search only.
    """

    def __init__(self, cfg: ElasticsearchConfig):
        if not cfg.cloud_id and not cfg.hosts:
            raise ValueError("Either cloud_id or hosts must be provided")
        if not cfg.api_key:
            raise ValueError("Missing ELASTICSEARCH_API_KEY")
        
        self.cfg = cfg
        
        # Initialize Elasticsearch client
        if cfg.cloud_id:
            self._client = AsyncElasticsearch(
                cloud_id=cfg.cloud_id,
                api_key=cfg.api_key,
            )
        else:
            self._client = AsyncElasticsearch(
                hosts=cfg.hosts,
                api_key=cfg.api_key,
            )

    @classmethod
    def from_env(cls) -> "ElasticsearchStore":
        """Create ElasticsearchStore from environment variables."""
        return cls(
            ElasticsearchConfig(
                cloud_id=os.getenv("ELASTICSEARCH_CLOUD_ID", ""),
                hosts=os.getenv("ELASTICSEARCH_HOSTS", "").split(",") if os.getenv("ELASTICSEARCH_HOSTS") else None,
                api_key=os.getenv("ELASTICSEARCH_API_KEY", ""),
                index_name=os.getenv("ELASTICSEARCH_INDEX_NAME", "patents"),
            )
        )

    async def close(self):
        """Close the Elasticsearch client connection."""
        await self._client.close()

    async def index_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
    ) -> None:
        """
        Index a single document.
        
        Args:
            doc_id: Document ID
            document: Document to index
        """
        log.debug(f"[ELASTICSEARCH] Indexing document: {doc_id}")
        await self._client.index(
            index=self.cfg.index_name,
            id=doc_id,
            document=document,
        )
        log.debug(f"[ELASTICSEARCH] Document indexed: {doc_id}")

    async def bulk_index(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """
        Bulk index documents.
        
        Args:
            documents: List of documents with '_id' field
        """
        from elasticsearch.helpers import async_bulk
        
        log.info(f"[ELASTICSEARCH] Bulk indexing {len(documents)} documents")
        
        actions = [
            {
                "_index": self.cfg.index_name,
                "_id": doc["_id"],
                "_source": {k: v for k, v in doc.items() if k != "_id"},
            }
            for doc in documents
        ]
        
        await async_bulk(self._client, actions)
        log.info(f"[ELASTICSEARCH] Bulk index complete: {len(documents)} documents")

    async def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 search on patent-level documents.
        
        Args:
            query_text: Query text for BM25 search
            top_k: Number of results to return
            metadata_filter: Optional metadata filters
            search_fields: Fields to search (default: ["title", "abstract", "claims"])
            
        Returns:
            List of search results with id, score, and metadata
        """
        if not query_text:
            raise ValueError("query_text is required")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        
        log.info(f"[ELASTICSEARCH] BM25 search (top_k={top_k})")
        log.debug(f"[ELASTICSEARCH] Query text: '{query_text[:100]}...'")
        
        # Default search fields for patents
        if search_fields is None:
            search_fields = ["title^2", "patent_id"]
        
        log.debug(f"[ELASTICSEARCH] Search fields: {search_fields}")
        
        # Build query
        query: Dict[str, Any] = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": search_fields,
                            "type": "best_fields",
                        }
                    }
                ]
            }
        }
        
        # Add metadata filters
        if metadata_filter:
            log.debug(f"[ELASTICSEARCH] Applying filters: {metadata_filter}")
            filter_clauses = self._build_filter_clauses(metadata_filter)
            if filter_clauses:
                query["bool"]["filter"] = filter_clauses
        
        # Execute search
        response = await self._client.search(
            index=self.cfg.index_name,
            query=query,
            size=top_k,
            _source=True,
        )
        
        # Format results
        results: List[Dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "metadata": hit["_source"],
            })
        
        log.info(f"[ELASTICSEARCH] BM25 search complete: found {len(results)} results")
        if results:
            log.debug(f"[ELASTICSEARCH] Top result score: {results[0]['score']:.4f}")
        
        return results

    def _build_filter_clauses(self, metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build Elasticsearch filter clauses from metadata filter.
        
        Args:
            metadata_filter: Metadata filter dictionary
            
        Returns:
            List of filter clauses
        """
        filter_clauses: List[Dict[str, Any]] = []
        
        for key, value in metadata_filter.items():
            if isinstance(value, dict):
                # Handle range queries ($gte, $lte, etc.)
                if "$gte" in value or "$lte" in value or "$gt" in value or "$lt" in value:
                    range_filter: Dict[str, Any] = {}
                    if "$gte" in value:
                        range_filter["gte"] = value["$gte"]
                    if "$lte" in value:
                        range_filter["lte"] = value["$lte"]
                    if "$gt" in value:
                        range_filter["gt"] = value["$gt"]
                    if "$lt" in value:
                        range_filter["lt"] = value["$lt"]
                    
                    filter_clauses.append({"range": {key: range_filter}})
                
                # Handle $in queries
                elif "$in" in value:
                    filter_clauses.append({"terms": {key: value["$in"]}})
            else:
                # Simple equality
                filter_clauses.append({"term": {key: value}})
        
        return filter_clauses

    async def create_index(
        self,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create the index with optional mappings and settings.
        
        Args:
            mappings: Index mappings
            settings: Index settings
        """
        # Default settings optimized for BM25
        if settings is None:
            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "patent_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_",
                        }
                    }
                }
            }
        
        # Default mappings for patent documents
        if mappings is None:
            mappings = {
                "properties": {
                    "patent_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "patent_analyzer"},
                    "cpc": {"type": "keyword"},
                    "filing_date": {"type": "date"},
                    "filing_year": {"type": "integer"},
                }
            }
                    
        body = {}
        if settings:
            body["settings"] = settings
        if mappings:
            body["mappings"] = mappings
        
        await self._client.indices.create(
            index=self.cfg.index_name,
            body=body,
        )

    async def delete_index(self) -> None:
        """Delete the index."""
        await self._client.indices.delete(index=self.cfg.index_name)

    async def index_exists(self) -> bool:
        """Check if the index exists."""
        return await self._client.indices.exists(index=self.cfg.index_name)
