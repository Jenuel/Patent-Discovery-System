from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from app.services.ingestion.parse import ParsedPatent


@dataclass(frozen=True)
class ChunkRecord:
    """
    One vector DB record (patent-level, claim-level, or limitation-level).
    """
    id: str
    level: str                 #
    patent_id: str
    claim_no: Optional[int]
    text: str
    snippet: str
    metadata: Dict[str, Any]


def _stable_id(*parts: str) -> str:
    s = "||".join(p.strip() for p in parts if p is not None)
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _snippet(text: str, n: int = 300) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= n else t[: n - 3] + "..."


def _parse_claim_no(claim_text: str) -> Optional[int]:
    m = re.match(r"^\s*(\d+)\.\s+", claim_text)
    return int(m.group(1)) if m else None


def chunk_patent_level(p: ParsedPatent) -> ChunkRecord:
    text = "\n".join([x for x in [p.title, p.abstract] if x]).strip()
    rid = _stable_id(p.patent_id, "patent")
    cpc_prefix = next((c[:4] for c in p.cpc_codes if len(c) >= 4), None)  
    md = {
        "chunk_id": rid,
        "patent_id": p.patent_id,
        "level": "patent",
        "title": p.title,
        "year": p.year,
        "cpc_codes": p.cpc_codes,
        "cpc_prefix": cpc_prefix,
    }
    return ChunkRecord(
        id=rid,
        level="patent",
        patent_id=p.patent_id,
        claim_no=None,
        text=text,
        snippet=_snippet(text),
        metadata=md,
    )


def chunk_claim_level(p: ParsedPatent, *, max_claims: Optional[int] = None) -> Iterator[ChunkRecord]:
    count = 0
    for claim in p.claims:
        if max_claims is not None and count >= max_claims:
            break
        claim = (claim or "").strip()
        if not claim:
            continue
        claim_no = _parse_claim_no(claim)
        rid = _stable_id(p.patent_id, "claim", str(claim_no or count))
        cpc_prefix = next((c[:4] for c in p.cpc_codes if len(c) >= 4), None)
        md = {
            "chunk_id": rid,
            "patent_id": p.patent_id,
            "level": "claim",
            "title": p.title,
            "claim_no": claim_no,
            "year": p.year,
            "cpc_codes": p.cpc_codes,
            "cpc_prefix": cpc_prefix,
            "text": claim,
            "snippet": _snippet(claim),
        }
        yield ChunkRecord(
            id=rid,
            level="claim",
            patent_id=p.patent_id,
            claim_no=claim_no,
            text=claim,
            snippet=_snippet(claim),
            metadata=md,
        )
        count += 1


def _split_limitations(claim_text: str) -> List[str]:
    """
    Naive limitation splitter:
    - Splits on semicolons
    - Further splits on "wherein", "whereby", "and" (lightly)
    Keep it conservative for patents—over-splitting can harm retrieval.
    """
    t = (claim_text or "").strip()
    if not t:
        return []

    t = re.sub(r"^\s*\d+\.\s+", "", t).strip()

    parts = [p.strip() for p in t.split(";") if p.strip()]
    out: List[str] = []
    for part in parts:
        sub = re.split(r"\b(wherein|whereby)\b", part, flags=re.IGNORECASE)
        if len(sub) > 1:
            buf = ""
            for s in sub:
                if s.lower() in ("wherein", "whereby"):
                    if buf.strip():
                        out.append(buf.strip())
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf.strip():
                out.append(buf.strip())
        else:
            out.append(part)

    seen = set()
    deduped: List[str] = []
    for x in out:
        key = re.sub(r"\s+", " ", x).strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(x)
    return deduped


def chunk_limitation_level(
    p: ParsedPatent,
    *,
    max_claims: Optional[int] = 10,
    max_limitations_per_claim: Optional[int] = 20,
) -> Iterator[ChunkRecord]:
    """
    Turn each claim into multiple limitation chunks.
    Useful for infringement-like matching.
    """
    claim_count = 0
    for claim in p.claims:
        if max_claims is not None and claim_count >= max_claims:
            break
        claim = (claim or "").strip()
        if not claim:
            continue
        claim_no = _parse_claim_no(claim)

        limitations = _split_limitations(claim)
        if max_limitations_per_claim is not None:
            limitations = limitations[:max_limitations_per_claim]

        for idx, lim in enumerate(limitations):
            rid = _stable_id(p.patent_id, "limitation", str(claim_no or claim_count), str(idx))
            cpc_prefix = next((c[:4] for c in p.cpc_codes if len(c) >= 4), None)
            md = {
                "chunk_id": rid,
                "patent_id": p.patent_id,
                "level": "limitation",
                "title": p.title,
                "claim_no": claim_no,
                "limitation_idx": idx,
                "year": p.year,
                "cpc_codes": p.cpc_codes,
                "cpc_prefix": cpc_prefix,
                "text": lim,
                "snippet": _snippet(lim),
            }
            yield ChunkRecord(
                id=rid,
                level="limitation",
                patent_id=p.patent_id,
                claim_no=claim_no,
                text=lim,
                snippet=_snippet(lim),
                metadata=md,
            )

        claim_count += 1
