from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import anyio
from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    """
    Env vars:
      OPENAI_API_KEY
      OPENAI_MODEL               (default: gpt-4.1-mini)
      OPENAI_MAX_OUTPUT_TOKENS   (default: 800)
      OPENAI_LLM_MAX_RETRIES     (default: 3)
      OPENAI_LLM_BACKOFF_BASE    (default: 0.7)
    """
    api_key: str
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 800
    max_retries: int = 3
    backoff_base_seconds: float = 0.7


Message = Dict[str, Any]  


class OpenAIClient:
    """
    LLM client wrapper (Responses API):
      - generate_text(prompt)
      - generate_chat(messages)

    Uses thread offloading so it’s safe in async FastAPI routes.
    """

    def __init__(self, cfg: LLMConfig):
        if not cfg.api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        if cfg.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0")
        if cfg.max_retries <= 0:
            raise ValueError("max_retries must be > 0")

        self.cfg = cfg
        self._client = OpenAI(api_key=cfg.api_key)

    @classmethod
    def from_env(cls) -> "OpenAIClient":
        return cls(
            LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "800")),
                max_retries=int(os.getenv("OPENAI_LLM_MAX_RETRIES", "3")),
                backoff_base_seconds=float(os.getenv("OPENAI_LLM_BACKOFF_BASE", "0.7")),
            )
        )

    async def generate_text(
        self,
        *,
        instructions: Optional[str] = None,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Simple “single prompt” generation.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be non-empty.")

        return await self._with_retries(
            lambda: self._responses_create_sync(
                instructions=instructions,
                input_data=prompt,
                metadata=metadata,
            )
        )

    async def generate_chat(
        self,
        *,
        messages: Sequence[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Multi-turn format (conversation-state style).
        messages example:
          [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        """
        if not messages:
            raise ValueError("messages must be non-empty")

        return await self._with_retries(
            lambda: self._responses_create_sync(
                instructions=None,
                input_data=list(messages),
                metadata=metadata,
            )
        )

    async def _with_retries(self, fn) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return await anyio.to_thread.run_sync(fn)
            except Exception as e:
                last_err = e
                await anyio.sleep(self.cfg.backoff_base_seconds * (2 ** (attempt - 1)))
        raise RuntimeError("OpenAI generation failed after retries") from last_err

    def _responses_create_sync(
        self,
        *,
        instructions: Optional[str],
        input_data: Union[str, List[Message]],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """
        Low-level Responses API call.
        Returns response.output_text (SDK convenience) when available. :contentReference[oaicite:2]{index=2}
        """
        resp = self._client.responses.create(
            model=self.cfg.model,
            instructions=instructions,
            input=input_data,
            max_output_tokens=self.cfg.max_output_tokens,
            metadata=metadata,
        )

        out_text = getattr(resp, "output_text", None)
        if out_text is not None:
            return out_text

        text_parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    text_parts.append(getattr(c, "text", ""))
        return "\n".join([t for t in text_parts if t])
