from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagPolicy:
    final_top_n: int = 8


DEFAULT_POLICY = RagPolicy(final_top_n=8)
