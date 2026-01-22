from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from app.api.v1.schemas.results import EvidenceItem
from app.services.llm.client import OpenAIClient


class Reranker(Protocol):
    async def rerank(self, query: str, items: Sequence[EvidenceItem]) -> List[EvidenceItem]:
        ...


@dataclass(frozen=True)
class RerankConfig:
    """
    max_candidates: limit how many retrieved items you ask the reranker to consider
    top_n: how many items to return after reranking
    snippet_chars: truncate evidence text for the reranker prompt (saves tokens + money)
    """
    max_candidates: int = 50
    top_n: int = 15
    snippet_chars: int = 900


def _make_snippet(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


class NoopReranker:
    """Keeps the original order."""
    def __init__(self, cfg: Optional[RerankConfig] = None):
        self.cfg = cfg or RerankConfig()

    async def rerank(self, query: str, items: Sequence[EvidenceItem]) -> List[EvidenceItem]:
        limited = list(items)[: self.cfg.max_candidates]
        return limited[: self.cfg.top_n]


class OpenAIReranker:
    """
    LLM-based reranker:
      - Takes top-K retrieved candidates
      - Asks the model to produce an ordered list of chunk_ids (or ids)
      - Reorders candidates accordingly

    Notes:
      - This is not as strong/cheap as a real cross-encoder reranker,
        but it's easy to implement and works well for prototypes.
      - For patents, reranking claim-level chunks is often the best ROI.
    """

    def __init__(
        self,
        llm: Optional[OpenAIClient] = None,
        cfg: Optional[RerankConfig] = None,
    ):
        self.llm = llm or OpenAIClient.from_env()
        self.cfg = cfg or RerankConfig()

    async def rerank(self, query: str, items: Sequence[EvidenceItem]) -> List[EvidenceItem]:
        candidates = list(items)[: self.cfg.max_candidates]
        if len(candidates) <= 1:
            return candidates

        prompt = self._build_prompt(query, candidates)

        instructions = (
            "You are a reranking model. Reorder candidates by relevance to the user query.\n"
            "Return ONLY valid JSON with this exact shape:\n"
            '{"ranked_ids": ["<id1>", "<id2>", "..."]}\n'
            "Rules:\n"
            "- Use candidate 'id' values exactly as given.\n"
            "- Include each id at most once.\n"
            "- If uncertain, keep original relative order.\n"
        )

        text = await self.llm.generate_text(instructions=instructions, prompt=prompt)

        ranked_ids = self._parse_ranked_ids(text)
        if not ranked_ids:
            return candidates[: self.cfg.top_n]

        by_id = {self._candidate_id(c): c for c in candidates}
        out: List[EvidenceItem] = []

        for rid in ranked_ids:
            c = by_id.get(rid)
            if c is not None:
                out.append(c)

        already = set(self._candidate_id(x) for x in out)
        for c in candidates:
            cid = self._candidate_id(c)
            if cid not in already:
                out.append(c)

        return out[: self.cfg.top_n]

    def _candidate_id(self, item: EvidenceItem) -> str:
        return item.chunk_id or f"{item.patent_id}:{item.level}:{item.claim_no}"

    def _build_prompt(self, query: str, candidates: Sequence[EvidenceItem]) -> str:
        lines: List[str] = []
        lines.append("User query:")
        lines.append(query.strip())
        lines.append("")
        lines.append("Candidates (rerank by relevance):")

        for idx, c in enumerate(candidates, 1):
            cid = self._candidate_id(c)
            title = c.title or ""
            claim = "" if c.claim_no is None else f"claim_no={c.claim_no}"
            snippet = _make_snippet(c.text, self.cfg.snippet_chars)

            lines.append(f"\n[{idx}] id={cid}")
            lines.append(f"patent_id={c.patent_id} level={c.level} {claim}".strip())
            if title:
                lines.append(f"title={title}")
            lines.append("text:")
            lines.append(snippet)

        lines.append("\nReturn JSON only.")
        return "\n".join(lines)

    def _parse_ranked_ids(self, raw: str) -> List[str]:
        """
        Attempts to parse JSON output robustly.
        Accepts:
          {"ranked_ids": [...]}
        """
        s = (raw or "").strip()

        obj_text = None
        if s.startswith("{") and s.endswith("}"):
            obj_text = s
        else:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj_text = s[start : end + 1]

        if not obj_text:
            return []

        try:
            data = json.loads(obj_text)
        except Exception:
            return []

        ranked = data.get("ranked_ids")
        if not isinstance(ranked, list):
            return []

        out: List[str] = []
        for x in ranked:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
