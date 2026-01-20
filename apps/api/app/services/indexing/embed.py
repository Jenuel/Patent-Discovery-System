from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Sequence, Optional
from openai import OpenAI 


@dataclass(frozen=True)
class EmbedConfig:
    """
    Environment variables:
      OPENAI_API_KEY
      OPENAI_EMBEDDING_MODEL         (default: text-embedding-3-small)
      OPENAI_EMBED_BATCH_SIZE        (default: 64)
      OPENAI_EMBED_MAX_RETRIES       (default: 5)
      OPENAI_EMBED_BACKOFF_BASE      (default: 0.7)
    """
    api_key: str
    model: str = "text-embedding-3-small"
    batch_size: int = 64
    max_retries: int = 5
    backoff_base_seconds: float = 0.7


class OpenAIEmbedder:
    """
    Clean embedding wrapper for BOTH:
      - offline indexing scripts
      - online retrieval (FastAPI)

    Methods:
      - embed(text) -> vector
      - embed_batch(texts) -> vectors (same order as input)
    """

    def __init__(self, cfg: EmbedConfig):
        if not cfg.api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if cfg.max_retries <= 0:
            raise ValueError("max_retries must be > 0")
        if cfg.backoff_base_seconds <= 0:
            raise ValueError("backoff_base_seconds must be > 0")

        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key)

    @classmethod
    def from_env(cls) -> "OpenAIEmbedder":
        return cls(
            EmbedConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                batch_size=int(os.getenv("OPENAI_EMBED_BATCH_SIZE", "64")),
                max_retries=int(os.getenv("OPENAI_EMBED_MAX_RETRIES", "5")),
                backoff_base_seconds=float(os.getenv("OPENAI_EMBED_BACKOFF_BASE", "0.7")),
            )
        )

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        cleaned = [self._clean_text(t) for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("One or more inputs are empty after cleaning.")

        out: List[List[float]] = []
        bs = self.cfg.batch_size

        for i in range(0, len(cleaned), bs):
            batch = cleaned[i : i + bs]
            out.extend(self._embed_batch_with_retries(batch))

        return out

    def _embed_batch_with_retries(self, batch: List[str]) -> List[List[float]]:
        last_err: Optional[Exception] = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.client.embeddings.create(
                    model=self.cfg.model,
                    input=batch,
                )
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                sleep_s = self.cfg.backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_s)

        raise RuntimeError("OpenAI embedding failed after retries.") from last_err

    @staticmethod
    def _clean_text(text: str) -> str:
        t = (text or "").strip()
        t = " ".join(t.split())  
        return t


_default_embedder: Optional[OpenAIEmbedder] = None


def get_embedder() -> OpenAIEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = OpenAIEmbedder.from_env()
    return _default_embedder


def embed_text(text: str) -> List[float]:
    return get_embedder().embed(text)


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    return get_embedder().embed_batch(texts)
