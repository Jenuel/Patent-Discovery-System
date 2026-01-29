from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.api.v1.schemas.results import EvidenceItem, QueryResponse
from app.services.indexing.embed import OpenAIEmbedder
from app.services.indexing.pinecone import PineconeStore
from app.services.indexing.schemas import SparseVector
from app.services.llm.client import OpenAIClient
from app.services.rag.policies import RagPolicy, DEFAULT_POLICY
from app.services.rerank.reranker import OpenAIReranker, RerankConfig
from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.fusion import ScoredMatch
from app.services.retrieval.hierarchical import HierarchicalRetriever, HierarchicalConfig
from app.services.retrieval.sparse import SparseRetriever


class RAGOrchestrator:
    """
    Main RAG orchestrator that combines all services:
    - Embedding (OpenAIEmbedder)
    - Vector storage (PineconeStore)
    - Retrieval (Dense, Sparse, Hierarchical)
    - Reranking (OpenAIReranker)
    - LLM generation (OpenAIClient)
    
    This orchestrates the full RAG pipeline for patent discovery.
    """

    def __init__(
        self,
        embedder: Optional[OpenAIEmbedder] = None,
        store: Optional[PineconeStore] = None,
        llm: Optional[OpenAIClient] = None,
        reranker: Optional[OpenAIReranker] = None,
        policy: Optional[RagPolicy] = None,
        hierarchical_config: Optional[HierarchicalConfig] = None,
        rerank_config: Optional[RerankConfig] = None,
    ):
        """
        Initialize the RAG orchestrator with all required services.
        
        Args:
            embedder: OpenAI embedder for query encoding
            store: Pinecone vector store
            llm: OpenAI client for answer generation
            reranker: LLM-based reranker
            policy: RAG policy configuration
            hierarchical_config: Configuration for hierarchical retrieval
            rerank_config: Configuration for reranking
        """
        # Initialize core services
        self.embedder = embedder or OpenAIEmbedder.from_env()
        self.store = store or PineconeStore.from_env()
        self.llm = llm or OpenAIClient.from_env()
        
        # Initialize retrievers
        self.dense_retriever = DenseRetriever(self.store)
        self.sparse_retriever = SparseRetriever(self.store)
        
        # Initialize hierarchical retriever
        self.hierarchical_config = hierarchical_config or HierarchicalConfig()
        self.hierarchical_retriever = HierarchicalRetriever(
            dense=self.dense_retriever,
            sparse=self.sparse_retriever,
            cfg=self.hierarchical_config,
        )
        
        # Initialize reranker
        self.rerank_config = rerank_config or RerankConfig()
        self.reranker = reranker or OpenAIReranker(llm=self.llm, cfg=self.rerank_config)
        
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
        use_hierarchical: bool = True,
        use_reranking: bool = True,
        sparse_query_vec: Optional[SparseVector] = None,
    ) -> QueryResponse:
        """
        Execute full RAG pipeline for a patent query.
        
        Args:
            query: User query string
            mode: Query mode (prior_art, infringement, landscape)
            metadata_filter: Optional metadata filters for retrieval
            use_hierarchical: Whether to use hierarchical retrieval
            use_reranking: Whether to apply reranking
            sparse_query_vec: Optional sparse vector for hybrid search
            
        Returns:
            QueryResponse with answer and evidence
        """
        # Step 1: Encode query
        dense_query_vec = await self._encode_query(query)
        
        # Step 2: Retrieve candidates
        candidates = await self._retrieve(
            dense_query_vec=dense_query_vec,
            sparse_query_vec=sparse_query_vec,
            metadata_filter=metadata_filter or {},
            use_hierarchical=use_hierarchical,
        )
        
        # Step 3: Convert to evidence items
        evidence_items = self._to_evidence_items(candidates, source="hybrid")
        
        # Step 4: Rerank if enabled
        if use_reranking and evidence_items:
            evidence_items = await self.reranker.rerank(query, evidence_items)
        
        # Step 5: Apply final policy (top-N)
        evidence_items = evidence_items[: self.policy.final_top_n]
        
        # Step 6: Generate answer
        answer = await self._generate_answer(query, evidence_items, mode)
        
        return QueryResponse(
            mode=mode,
            answer=answer,
            evidence=evidence_items,
        )

    async def _encode_query(self, query: str) -> List[float]:
        """
        Encode query text to dense vector.
        """
        import anyio
        return await anyio.to_thread.run_sync(lambda: self.embedder.embed(query))

    async def _retrieve(
        self,
        dense_query_vec: List[float],
        sparse_query_vec: Optional[SparseVector],
        metadata_filter: Dict[str, Any],
        use_hierarchical: bool,
    ) -> List[ScoredMatch]:
        """
        Retrieve candidates using configured retrieval strategy.
        """
        if use_hierarchical:
            return await self.hierarchical_retriever.retrieve_claims_hierarchical(
                dense_query_vec=dense_query_vec,
                sparse_query_vec=sparse_query_vec,
                base_filter=metadata_filter,
            )
        else:
            results = await self.dense_retriever.search(
                dense_vector=dense_query_vec,
                top_k=50,
                metadata_filter=metadata_filter,
            )
            from app.services.retrieval.fusion import to_scored_matches
            return to_scored_matches(results)

    def _to_evidence_items(
        self,
        matches: List[ScoredMatch],
        source: str,
    ) -> List[EvidenceItem]:
        """
        Convert ScoredMatch objects to EvidenceItem schema.
        """
        items: List[EvidenceItem] = []
        
        for match in matches:
            metadata = match.metadata
            
            items.append(
                EvidenceItem(
                    chunk_id=metadata.get("chunk_id", match.id),
                    patent_id=metadata.get("patent_id", ""),
                    level=metadata.get("level", "claim"),
                    title=metadata.get("title"),
                    claim_no=metadata.get("claim_no"),
                    text=metadata.get("text", metadata.get("snippet", "")),
                    score=match.score,
                    source=source,
                    metadata=metadata,
                )
            )
        
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
            return "No relevant patents found for your query."
        
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
        
        answer = await self.llm.generate_text(
            instructions=instructions,
            prompt=prompt,
        )
        
        return answer

    async def retrieve_only(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        use_hierarchical: bool = True,
        sparse_query_vec: Optional[SparseVector] = None,
    ) -> List[EvidenceItem]:
        """
        Retrieve evidence without generating an answer.
        Useful for debugging or custom processing.
        """
        dense_query_vec = await self._encode_query(query)
        
        candidates = await self._retrieve(
            dense_query_vec=dense_query_vec,
            sparse_query_vec=sparse_query_vec,
            metadata_filter=metadata_filter or {},
            use_hierarchical=use_hierarchical,
        )
        
        evidence_items = self._to_evidence_items(candidates, source="hybrid")
        return evidence_items[:top_k]
