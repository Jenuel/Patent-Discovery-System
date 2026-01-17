from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field
import os

class Settings(BaseModel):
    env: str = Field(default="dev")
    cors_allow_origins: List[str] = Field(default_factory=list)

    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None
    openai_api_key: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = os.getenv("CORS_ALLOW_ORIGINS", "")
    cors_allow_origins = [o.strip() for o in origins.split(",") if o.strip()]

    return Settings(
        env=os.getenv("ENV", "dev"),
        cors_allow_origins=cors_allow_origins,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index=os.getenv("PINECONE_INDEX"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
