from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from app.api.v1.schemas.query import QueryRequest
from app.api.v1.schemas.results import QueryResponse, ErrorResponse
from app.core.logging import get_logger
from app.services.rag.orchestrator import RAGOrchestrator

router = APIRouter()
log = get_logger(__name__)


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Main endpoint for:
    - Prior art search
    - Infringement-style element matching (if user provides system_description / mentions infringement)
    - Landscape summaries (if query asks for summary/trends)

    """
    try:
        orchestrator = RAGOrchestrator.from_env()
        mode = _determine_mode(req)
        metadata_filter = _build_metadata_filter(req)
        
        response = await orchestrator.query(
            query=req.query,
            mode=mode,
            metadata_filter=metadata_filter,
            use_hierarchical=True,
            use_reranking=True,
            use_sparse=True,  # Use Elasticsearch BM25 for sparse retrieval
        )
        
        log.info(
            f"Query processed successfully: mode={mode}, "
            f"evidence_count={len(response.evidence)}"
        )
        
        return response
        
    except ValueError as e:
        log.warning(f"Invalid query request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e


def _determine_mode(req: QueryRequest) -> str:
    """
    Determine the query mode based on request content.
    
    Args:
        req: The query request
        
    Returns:
        Query mode: 'prior_art', 'infringement', or 'landscape'
    """
    query_lower = req.query.lower()
    
    if req.system_description or "infringement" in query_lower or "infringe" in query_lower:
        return "infringement"
    
    landscape_keywords = ["landscape", "summary", "trend", "overview", "analysis"]
    if any(keyword in query_lower for keyword in landscape_keywords):
        return "landscape"

    return "prior_art"


def _build_metadata_filter(req: QueryRequest) -> Dict[str, Any]:
    """
    Build Pinecone metadata filter from query request filters.
    
    Args:
        req: The query request
        
    Returns:
        Metadata filter dictionary for Pinecone
    """
    if not req.filters:
        return {}
    
    metadata_filter: Dict[str, Any] = {}

    if req.filters.cpc_prefixes:
        metadata_filter["cpc"] = {"$in": req.filters.cpc_prefixes}
    
    if req.filters.year_from or req.filters.year_to:
        year_filter: Dict[str, int] = {}
        if req.filters.year_from:
            year_filter["$gte"] = req.filters.year_from
        if req.filters.year_to:
            year_filter["$lte"] = req.filters.year_to
        metadata_filter["year"] = year_filter
    
    if req.filters.assignees:
        metadata_filter["assignee"] = {"$in": req.filters.assignees}
    
    return metadata_filter
