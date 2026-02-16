from typing import Any, Dict, List

from app.services.indexing.elasticsearch import ElasticsearchStore


class SparseRetriever:
    """
    Responsible for lexical retrieval using Elasticsearch BM25.
    Operates at PATENT LEVEL ONLY.
    """

    def __init__(self, store: ElasticsearchStore):
        self.store = store

    async def search(
        self,
        query_text: str,
        top_k: int,
        metadata_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Pure BM25 sparse retrieval at patent level using Elasticsearch.
        
        Args:
            query_text: Query text for BM25 search
            top_k: Number of results to return
            metadata_filter: Metadata filters (must include level='patent')
            
        Returns:
            List of patent-level search results
        """
        # Ensure we're only searching at patent level
        # The filter should already contain level='patent' from hierarchical retriever
        return await self.store.search_bm25(
            query_text=query_text,
            top_k=top_k,
            metadata_filter=metadata_filter,
            search_fields=["title^2", "abstract", "text", "claims", "patent_id"],
        )
