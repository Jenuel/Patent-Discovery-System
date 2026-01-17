from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.api.v1.schemas.query import QueryRequest
from app.api.v1.schemas.results import QueryResponse, ErrorResponse
from app.core.logging import get_logger

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
        # TODO: implement query logic
        return QueryResponse(
            mode="prior_art",
            answer="",
            evidence=[],
        )
    except ValueError as e:
        # validation / bad user input
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e
