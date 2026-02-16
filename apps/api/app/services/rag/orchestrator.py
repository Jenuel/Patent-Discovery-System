from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

from app.api.v1.schemas.results import EvidenceItem, QueryResponse
from app.services.indexing.embed import OpenAIEmbedder
from app.services.indexing.pinecone import PineconeStore
from app.services.indexing.elasticsearch import ElasticsearchStore
from app.services.llm.client import GeminiClient
from app.services.rag.policies import RagPolicy, DEFAULT_POLICY
# from app.services.rerank.reranker import GeminiReranker, RerankConfig  # TEMPORARILY DISABLED
from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.fusion import ScoredMatch
from app.services.retrieval.hierarchical import HierarchicalRetriever, HierarchicalConfig
from app.services.retrieval.sparse import SparseRetriever
from app.services.storage.mongodb import MongoDBStore

log = get_logger(__name__)


class RAGOrchestrator:
    """
    Main RAG orchestrator that combines all services:
    - Embedding (OpenAIEmbedder)
    - Dense vector storage (PineconeStore) - for semantic search
    - Sparse search (ElasticsearchStore) - for BM25 lexical search
    - Retrieval (Dense via Pinecone, Sparse via Elasticsearch, Hierarchical fusion)
    - Reranking (GeminiReranker)
    - LLM generation (GeminiClient)
    
    This orchestrates the full RAG pipeline for patent discovery.
    """

    def __init__(
        self,
        embedder: Optional[OpenAIEmbedder] = None,
        pinecone_store: Optional[PineconeStore] = None,
        elasticsearch_store: Optional[ElasticsearchStore] = None,
        mongodb_store: Optional[MongoDBStore] = None,
        llm: Optional[GeminiClient] = None,
        # reranker: Optional[GeminiReranker] = None,  # TEMPORARILY DISABLED
        policy: Optional[RagPolicy] = None,
        hierarchical_config: Optional[HierarchicalConfig] = None,
        # rerank_config: Optional[RerankConfig] = None,  # TEMPORARILY DISABLED
    ):
        """
        Initialize the RAG orchestrator with all required services.
        
        Args:
            embedder: OpenAI embedder for query encoding
            pinecone_store: Pinecone vector store for dense retrieval
            elasticsearch_store: Elasticsearch store for sparse BM25 retrieval
            mongodb_store: MongoDB store for retrieving raw text content
            llm: Gemini client for answer generation
            # reranker: LLM-based reranker  # TEMPORARILY DISABLED
            policy: RAG policy configuration
            hierarchical_config: Configuration for hierarchical retrieval
            # rerank_config: Configuration for reranking  # TEMPORARILY DISABLED
        """
        # Initialize core services
        self.embedder = embedder or OpenAIEmbedder.from_env()
        self.pinecone_store = pinecone_store or PineconeStore.from_env()
        self.elasticsearch_store = elasticsearch_store or ElasticsearchStore.from_env()
        self.mongodb_store = mongodb_store or MongoDBStore.from_env()
        self.llm = llm or GeminiClient.from_env()
        
        # Initialize retrievers
        self.dense_retriever = DenseRetriever(self.pinecone_store)
        self.sparse_retriever = SparseRetriever(self.elasticsearch_store)
        
        # Initialize hierarchical retriever
        self.hierarchical_config = hierarchical_config or HierarchicalConfig()
        self.hierarchical_retriever = HierarchicalRetriever(
            dense=self.dense_retriever,
            sparse=self.sparse_retriever,
            cfg=self.hierarchical_config,
        )
        
        # Initialize reranker - TEMPORARILY DISABLED
        # self.rerank_config = rerank_config or RerankConfig()
        # self.reranker = reranker or GeminiReranker(llm=self.llm, cfg=self.rerank_config)
        
        # Policy
        self.policy = policy or DEFAULT_POLICY

    @classmethod
    def from_env(cls) -> "RAGOrchestrator":
        """
        Create orchestrator from environment variables.
        """
        return cls()

    async def query(
        self,
        query: str,
        mode: str = "prior_art",
        metadata_filter: Optional[Dict[str, Any]] = None,
        # use_reranking: bool = True,  # TEMPORARILY DISABLED
    ) -> QueryResponse:
        """
        Execute full RAG pipeline for a patent query.
        
        Args:
            query: User query string
            mode: Query mode (prior_art, infringement, landscape)
            metadata_filter: Optional metadata filters for retrieval
            # use_reranking: Whether to apply reranking  # TEMPORARILY DISABLED
            
        Returns:
            QueryResponse with answer and evidence
        """
        log.info(f"[ORCHESTRATOR] Starting RAG pipeline for mode: {mode}")
        
        # Step 1: Encode query
        log.info("[ORCHESTRATOR STEP 1/6] Encoding query to dense vector")
        dense_query_vec = await self._encode_query(query)
        log.info(f"[ORCHESTRATOR STEP 1/6] Query encoded, vector dimension: {len(dense_query_vec)}")
        
        # Step 2: Retrieve candidates
        log.info("[ORCHESTRATOR STEP 2/6] Retrieving candidates via hierarchical retrieval")
        candidates = await self._retrieve(
            query=query,
            dense_query_vec=dense_query_vec,
            metadata_filter=metadata_filter or {},
        )
        log.info(f"[ORCHESTRATOR STEP 2/6] Retrieved {len(candidates)} candidates")
        
        # Step 3: Convert to evidence items (fetch text from MongoDB)
        log.info("[ORCHESTRATOR STEP 3/6] Converting candidates to evidence items (fetching from MongoDB)")
        evidence_items = await self._to_evidence_items(candidates, source="hybrid")
        log.info(f"[ORCHESTRATOR STEP 3/6] Converted to {len(evidence_items)} evidence items")
        
        # Step 4: Rerank if enabled - TEMPORARILY DISABLED
        # if use_reranking and evidence_items:
        #     log.info("[ORCHESTRATOR STEP 4/6] Reranking evidence items")
        #     evidence_items = await self.reranker.rerank(query, evidence_items)
        #     log.info(f"[ORCHESTRATOR STEP 4/6] Reranked to {len(evidence_items)} items")
        
        # Step 5: Apply final policy (top-N)
        log.info(f"[ORCHESTRATOR STEP 4/6] Applying final policy (top-{self.policy.final_top_n})")
        evidence_items = evidence_items[: self.policy.final_top_n]
        log.info(f"[ORCHESTRATOR STEP 4/6] Final evidence count: {len(evidence_items)}")
        
        # Step 6: Generate answer
        log.info("[ORCHESTRATOR STEP 5/6] Generating answer using LLM")
        answer = await self._generate_answer(query, evidence_items, mode)
        log.info(f"[ORCHESTRATOR STEP 5/6] Answer generated, length: {len(answer)} chars")
        
        log.info("[ORCHESTRATOR STEP 6/6] Building final response")
        response = QueryResponse(
            mode=mode,
            answer=answer,
            evidence=evidence_items,
        )
        log.info("[ORCHESTRATOR] RAG pipeline completed successfully")
        
        return response

    async def _encode_query(self, query: str) -> List[float]:
        """
        Encode query text to dense vector.
        """
        log.debug(f"Encoding query: '{query[:100]}...'")
        import anyio
        vector = await anyio.to_thread.run_sync(lambda: self.embedder.embed(query))
        log.debug(f"Query encoded successfully, vector dimension: {len(vector)}")
        return vector

    async def _retrieve(
        self,
        query: str,
        dense_query_vec: List[float],
        metadata_filter: Dict[str, Any],
    ) -> List[ScoredMatch]:
        """
        Retrieve candidates using hierarchical retrieval with sparse (BM25) enabled.
        Always uses both dense (Pinecone) and sparse (Elasticsearch) retrieval.
        """
        log.debug(f"Starting hierarchical retrieval with filter: {metadata_filter}")
        candidates = await self.hierarchical_retriever.retrieve_claims_hierarchical(
            dense_query_vec=dense_query_vec,
            query_text=query,
            base_filter=metadata_filter,
        )
        log.debug(f"Hierarchical retrieval completed, found {len(candidates)} candidates")
        return candidates

    async def _to_evidence_items(
        self,
        matches: List[ScoredMatch],
        source: str,
    ) -> List[EvidenceItem]:
        """
        Convert ScoredMatch objects to EvidenceItem schema.
        Fetches raw text content from MongoDB using chunk IDs.
        
        Args:
            matches: List of scored matches from retrieval
            source: Source of the matches (dense|sparse|hybrid|reranked)
            
        Returns:
            List of EvidenceItem objects with text populated from MongoDB
        """
        if not matches:
            log.debug("No matches to convert to evidence items")
            return []
        
        # Extract all chunk IDs from matches
        chunk_ids = [match.id for match in matches]
        log.debug(f"Fetching {len(chunk_ids)} chunks from MongoDB")
        
        # Fetch all chunks from MongoDB in a single batch query
        chunks_map = await self.mongodb_store.get_chunks_by_ids(chunk_ids)
        log.debug(f"Retrieved {len(chunks_map)} chunks from MongoDB")
        
        items: List[EvidenceItem] = []
        
        for match in matches:
            metadata = match.metadata
            chunk_id = match.id
            
            # Get the chunk document from MongoDB
            chunk_doc = chunks_map.get(chunk_id, {})
            
            # Extract text from MongoDB document, fallback to metadata if not found
            text = chunk_doc.get("text", metadata.get("text", metadata.get("snippet", "")))
            
            items.append(
                EvidenceItem(
                    chunk_id=chunk_id,
                    patent_id=metadata.get("patent_id", chunk_doc.get("patent_id", "")),
                    level=metadata.get("level", chunk_doc.get("section", "claim")),
                    title=metadata.get("title", chunk_doc.get("title")),
                    claim_no=metadata.get("claim_no", chunk_doc.get("claim_no")),
                    text=text,
                    score=match.score,
                    source=source,
                    metadata=metadata,
                )
            )
        
        log.debug(f"Converted {len(items)} matches to evidence items")
        return items

    async def _generate_answer(
        self,
        query: str,
        evidence: List[EvidenceItem],
        mode: str,
    ) -> str:
        """
        Generate answer using LLM based on retrieved evidence.
        """
        if not evidence:
            log.warning("No evidence available for answer generation")
            return "No relevant patents found for your query."
        
        log.debug(f"Generating answer for {len(evidence)} evidence items")
        
        # Build context from evidence
        context_parts: List[str] = []
        for idx, item in enumerate(evidence, 1):
            context_parts.append(
                f"[{idx}] Patent: {item.patent_id} | Level: {item.level} | "
                f"Claim: {item.claim_no or 'N/A'}\n"
                f"Title: {item.title or 'N/A'}\n"
                f"Text: {item.text[:500]}...\n"
            )
        
        context = "\n".join(context_parts)
        log.debug(f"Built context with {len(context)} characters")
        
        # Mode-specific instructions
        mode_instructions = {
            "prior_art": (
                "You are a patent prior art search assistant. "
                "Analyze the evidence and identify relevant prior art patents. "
                "Explain how they relate to the query."
            ),
            "infringement": (
                "You are a patent infringement analysis assistant. "
                "Analyze the evidence and identify potential infringement issues. "
                "Explain which claims may be relevant."
            ),
            "landscape": (
                "You are a patent landscape analysis assistant. "
                "Analyze the evidence and provide an overview of the patent landscape. "
                "Identify key trends and technologies."
            ),
        }
        
        instructions = mode_instructions.get(
            mode,
            "You are a patent search assistant. Analyze the evidence and answer the query.",
        )
        
        prompt = (
            f"Query: {query}\n\n"
            f"Evidence:\n{context}\n\n"
            f"Based on the evidence above, provide a comprehensive answer to the query."
        )
        
        log.debug(f"Calling LLM with prompt length: {len(prompt)} characters")
        answer = await self.llm.generate_text(
            instructions=instructions,
            prompt=prompt,
        )
        log.debug(f"LLM response received, answer length: {len(answer)} characters")
        
        return answer

