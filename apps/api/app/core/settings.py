from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

class Settings(BaseModel):
    env: str = Field(default="dev")
    cors_allow_origins: List[str] = Field(default_factory=list)

    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    retrieval_arm: str = "hybrid"
    fusion_dense_weight: float = 0.9
    fusion_sparse_weight: float = 0.1


def _float_env(name: str, default: float) -> float:
    """Parse a float env var, failing loudly on a malformed value.

    ``float("")`` and ``float("0.9x")`` both raise, which is what we want: a
    typo'd weight should stop startup rather than silently fall back to a
    default and ship a ratio nobody chose.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    origins = os.getenv("CORS_ALLOW_ORIGINS", "")
    cors_allow_origins = [o.strip() for o in origins.split(",") if o.strip()]

    return Settings(
        env=os.getenv("ENV", "dev"),
        cors_allow_origins=cors_allow_origins,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        rerank_enabled=os.getenv("RERANK_ENABLED", "false").strip().lower()
        in ("true", "1", "yes"),
        rerank_model=os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        retrieval_arm=os.getenv("RETRIEVAL_ARM", "hybrid").strip().lower() or "hybrid",
        fusion_dense_weight=_float_env("FUSION_DENSE_WEIGHT", 0.9),
        fusion_sparse_weight=_float_env("FUSION_SPARSE_WEIGHT", 0.1),
    )
