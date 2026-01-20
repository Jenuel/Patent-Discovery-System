from __future__ import annotations

from typing import List

from app.api.v1.schemas.results import EvidenceItem


def build_prior_art_prompt(user_query: str, evidence: List[EvidenceItem]) -> str:
    blocks = []
    for i, e in enumerate(evidence, 1):
        head = f"[{i}] {e.patent_id}" + (f" — {e.title}" if e.title else "")
        blocks.append(f"{head}\nClaim: {e.claim_no}\n{e.text}")
    return (
        "You are a patent prior-art assistant.\n"
        "Use ONLY the evidence. Cite using [#].\n\n"
        f"User query:\n{user_query}\n\n"
        "Evidence:\n" + "\n\n".join(blocks) + "\n\n"
        "Task: Identify the most relevant prior art and explain why, referencing claim elements.\n"
    )
