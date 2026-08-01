from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

from app.api.v1.schemas.results import EvidenceItem, QueryResponse
from app.services.indexing.embed import OpenAIEmbedder
from app.services.indexing.qdrant import QdrantHybridStore
from app.services.llm.client import GeminiClient
from app.services.rag.policies import RagPolicy, DEFAULT_POLICY
from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.fusion import ScoredMatch
from app.services.retrieval.hierarchical import HierarchicalRetriever, HierarchicalConfig
from app.services.storage.mongodb import MongoDBStore

log = get_logger(__name__)


class RAGOrchestrator:
    """
    Main RAG orchestrator that combines all services:
    - Embedding (OpenAIEmbedder)
    - Vector storage (QdrantHybridStore) — patent-level hybrid + claim-level dense
    - Retrieval (HierarchicalRetriever → DenseRetriever → QdrantHybridStore)
    - LLM generation (GeminiClient)

    Dense and sparse patent retrieval are handled natively inside Qdrant
    via Prefetch + RRF fusion.  Claim-level retrieval uses the dedicated
    ``claims_hybrid`` Qdrant collection.
    """

    def __init__(
        self,
        embedder: Optional[OpenAIEmbedder] = None,
        qdrant_store: Optional[QdrantHybridStore] = None,
        mongodb_store: Optional[MongoDBStore] = None,
        llm: Optional[GeminiClient] = None,
        policy: Optional[RagPolicy] = None,
        hierarchical_config: Optional[HierarchicalConfig] = None,
        reranker: Optional[Any] = None,
    ):
        """
        Initialize the RAG orchestrator with all required services.

        Args:
            embedder:            OpenAI embedder for query encoding.
            qdrant_store:        Qdrant hybrid store for both patent and claim retrieval.
            mongodb_store:       MongoDB store for retrieving raw text content.
            llm:                 Gemini client for answer generation.
            policy:              RAG policy configuration.
            hierarchical_config: Configuration for hierarchical retrieval.
            reranker:            Cross-encoder reranker (RET-07). Built from
                                 settings when omitted; pass ``False`` to
                                 disable explicitly, or an instance to inject.
        """
        # Initialize core services
        self.embedder = embedder or OpenAIEmbedder.from_env()
        self.qdrant_store = qdrant_store or QdrantHybridStore.from_env()
        self.mongodb_store = mongodb_store or MongoDBStore.from_env()
        self.llm = llm or GeminiClient.from_env()

        # Initialize retrievers
        self.dense_retriever = DenseRetriever(self.qdrant_store)

        if reranker is None:
            from app.core.settings import get_settings

            if get_settings().rerank_enabled:
                from app.services.rerank.reranker import CrossEncoderReranker

                reranker = CrossEncoderReranker.from_env()
        self.reranker = reranker or None

        # Initialize hierarchical retriever
        self.hierarchical_config = hierarchical_config or HierarchicalConfig()
        self.hierarchical_retriever = HierarchicalRetriever(
            dense=self.dense_retriever,
            cfg=self.hierarchical_config,
            reranker=self.reranker,
        )

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
        top_k: Optional[int] = None,
    ) -> QueryResponse:
        """
        Execute full RAG pipeline for a patent query.

        Args:
            query: User query string
            mode: Query mode (prior_art, infringement, landscape)
            metadata_filter: Optional metadata filters for retrieval
            top_k: Override for the number of evidence items returned. Cannot
                exceed what Stage 2 retrieved, so callers asking for more than
                claim_top_k simply get everything available.

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
        
        # Step 4: Apply final policy (top-N)
        final_top_n = top_k if top_k is not None else self.policy.final_top_n
        log.info(f"[ORCHESTRATOR STEP 4/6] Applying final policy (top-{final_top_n})")
        evidence_items = evidence_items[:final_top_n]
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
        Retrieve candidates using hierarchical retrieval backed by Qdrant.
        Stage 1 uses Qdrant native hybrid (dense + BM25) on patents_hybrid.
        Stage 2 uses Qdrant dense search on claims_hybrid.
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
            chunk_id = match.id

            # Get the chunk document from MongoDB
            chunk_doc = chunks_map.get(chunk_id, {})

            metadata = chunk_doc.get("metadata") or {}

            def field(*names: str, default: Any = None) -> Any:
                for name in names:
                    for src in (metadata, chunk_doc, match.metadata):
                        value = src.get(name)
                        if value not in (None, ""):
                            return value
                return default

            text = field("raw_text", "text", "content", "snippet", default="")

            items.append(
                EvidenceItem(
                    chunk_id=chunk_id,
                    patent_id=field("patent_id", default=""),
                    level=field("section", "level", default="claim"),
                    title=field("title"),
                    # Chunk documents spell this claim_no; keep claim_number as
                    # an alias so either ingestion shape works.
                    claim_no=field("claim_no", "claim_number"),
                    text=text,
                    score=match.score,
                    source=source,
                    metadata=metadata or {
                        k: v for k, v in chunk_doc.items() if k != "_id"
                    } or match.metadata,
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

