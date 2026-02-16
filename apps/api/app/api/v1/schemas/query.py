from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class QueryFilters(BaseModel):
    cpc_prefixes: Optional[List[str]] = Field(default=None, description="e.g., ['G06N']")
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    assignees: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User query, e.g., prior art / infringement question")
    system_description: Optional[str] = Field(
        default=None,
        description="Optional: the user's product/system description for infringement-style matching",
    )
    filters: Optional[QueryFilters] = None
