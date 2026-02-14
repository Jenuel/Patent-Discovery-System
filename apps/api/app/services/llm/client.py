from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import anyio
import google.generativeai as genai


@dataclass(frozen=True)
class LLMConfig:
    """
    Env vars:
      GEMINI_API_KEY
      GEMINI_MODEL               (default: gemini-1.5-flash)
      GEMINI_MAX_OUTPUT_TOKENS   (default: 2048)
      GEMINI_LLM_MAX_RETRIES     (default: 3)
      GEMINI_LLM_BACKOFF_BASE    (default: 0.7)
      GEMINI_TEMPERATURE         (default: 0.7)
    """
    api_key: str
    model: str = "gemini-1.5-flash"
    max_output_tokens: int = 2048
    max_retries: int = 3
    backoff_base_seconds: float = 0.7
    temperature: float = 0.7


Message = Dict[str, Any]  


class GeminiClient:
    """
    LLM client wrapper for Google Gemini:
      - generate_text(prompt)
      - generate_chat(messages)

    Uses thread offloading so it's safe in async FastAPI routes.
    """

    def __init__(self, cfg: LLMConfig):
        if not cfg.api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        if cfg.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0")
        if cfg.max_retries <= 0:
            raise ValueError("max_retries must be > 0")

        self.cfg = cfg
        
        # Configure the Gemini API
        genai.configure(api_key=cfg.api_key)
        
        # Create generation config
        self.generation_config = genai.GenerationConfig(
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )
        
        # Initialize the model
        self._model = genai.GenerativeModel(
            model_name=cfg.model,
            generation_config=self.generation_config,
        )

    @classmethod
    def from_env(cls) -> "GeminiClient":
        return cls(
            LLMConfig(
                api_key=os.getenv("GEMINI_API_KEY", ""),
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048")),
                max_retries=int(os.getenv("GEMINI_LLM_MAX_RETRIES", "3")),
                backoff_base_seconds=float(os.getenv("GEMINI_LLM_BACKOFF_BASE", "0.7")),
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
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
        Simple "single prompt" generation.
        Combines instructions and prompt into a single message.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be non-empty.")

        return await self._with_retries(
            lambda: self._generate_sync(
                instructions=instructions,
                prompt=prompt,
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
        
        Gemini uses 'user' and 'model' roles instead of 'user' and 'assistant'.
        """
        if not messages:
            raise ValueError("messages must be non-empty")

        return await self._with_retries(
            lambda: self._generate_chat_sync(messages=messages)
        )

    async def _with_retries(self, fn) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return await anyio.to_thread.run_sync(fn)
            except Exception as e:
                last_err = e
                await anyio.sleep(self.cfg.backoff_base_seconds * (2 ** (attempt - 1)))
        raise RuntimeError("Gemini generation failed after retries") from last_err

    def _generate_sync(
        self,
        *,
        instructions: Optional[str],
        prompt: str,
    ) -> str:
        """
        Low-level Gemini API call for single prompt generation.
        """
        # Combine instructions and prompt
        full_prompt = prompt
        if instructions:
            full_prompt = f"{instructions}\n\n{prompt}"
        
        # Generate content
        response = self._model.generate_content(full_prompt)
        
        # Extract text from response
        if response.text:
            return response.text
        
        # Fallback: try to extract from parts
        if hasattr(response, 'parts') and response.parts:
            text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
            return "\n".join(text_parts)
        
        return ""

    def _generate_chat_sync(
        self,
        *,
        messages: Sequence[Message],
    ) -> str:
        """
        Low-level Gemini API call for chat-style generation.
        Converts OpenAI-style messages to Gemini format.
        """
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map OpenAI roles to Gemini roles
            gemini_role = "model" if role == "assistant" else "user"
            
            gemini_messages.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })
        
        # Start a chat session
        chat = self._model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        # Send the last message
        last_message = gemini_messages[-1]["parts"][0]["text"]
        response = chat.send_message(last_message)
        
        # Extract text from response
        if response.text:
            return response.text
        
        return ""
