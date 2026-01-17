from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    chunk_id: str
    patent_id: str
    level: str = Field(..., description="patent|claim|limitation")
    title: Optional[str] = None
    claim_no: Optional[int] = None
    text: str
    score: float
    source: str = Field(..., description="dense|sparse|hybrid|reranked")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    mode: str = Field(..., description="prior_art|infringement|landscape")
    answer: str
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
