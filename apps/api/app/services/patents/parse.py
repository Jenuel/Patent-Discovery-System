from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParsedPatent:
    patent_id: str
    title: str
    abstract: str
    claims: List[str]
    year: Optional[int]
    cpc_codes: List[str]


def _first_str(*candidates: Any) -> str:
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return ""


def _first_int(*candidates: Any) -> Optional[int]:
    for c in candidates:
        try:
            if c is None:
                continue
            if isinstance(c, bool):
                continue
            if isinstance(c, int):
                return c
            if isinstance(c, str) and c.strip():
                return int(re.findall(r"\d{4}", c)[0]) if re.findall(r"\d{4}", c) else None
        except Exception:
            continue
    return None


def extract_patent_id(raw: Dict) -> str:
    """
    Tries common HUPD-ish keys. Falls back to a stable hash-like string.
    """
    pid = _first_str(
        raw.get("publication_number"),
        raw.get("patent_id"),
        raw.get("application_number"),
        raw.get("app_id"),
        raw.get("number"),
        raw.get("id"),
    )
    if pid:
        return pid
    return f"UNKNOWN:{abs(hash(_first_str(raw.get('title'), raw.get('abstract'))))}"


def extract_title(raw: Dict) -> str:
    return _first_str(
        raw.get("title"),
        raw.get("invention_title"),
        (raw.get("bibliographic") or {}).get("title") if isinstance(raw.get("bibliographic"), dict) else None,
    )


def extract_abstract(raw: Dict) -> str:
    a = raw.get("abstract")
    if isinstance(a, str):
        return a.strip()

    if isinstance(a, dict):
        return _first_str(a.get("text"), a.get("abstract"))
    if isinstance(a, list):
        parts = [x.strip() for x in a if isinstance(x, str) and x.strip()]
        return "\n".join(parts)
    return ""


def extract_claims(raw: Dict) -> List[str]:
    """
    Accepts:
      - raw["claims"] as string
      - raw["claims"] as list[str]
      - raw["claims"] as list[dict] with keys like "text"
    Returns list of claim strings (one per claim).
    """
    c = raw.get("claims")

    if isinstance(c, str) and c.strip():
        text = c.strip()
        chunks = re.split(r"\n\s*(\d+)\.\s+", "\n" + text)
        if len(chunks) > 1:
            out: List[str] = []
            i = 1
            while i + 1 < len(chunks):
                num = chunks[i]
                body = chunks[i + 1].strip()
                if body:
                    out.append(f"{num}. {body}")
                i += 2
            return out
        return [text]

    if isinstance(c, list):
        out: List[str] = []
        for item in c:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                txt = _first_str(item.get("text"), item.get("claim"), item.get("content"))
                if txt:
                    num = item.get("claim_num") or item.get("number") or item.get("claim_number")
                    if num is not None:
                        out.append(f"{num}. {txt}")
                    else:
                        out.append(txt)
        return out

    claims_obj = raw.get("claims_text") or raw.get("claim_text") or raw.get("claim")
    if isinstance(claims_obj, str) and claims_obj.strip():
        return [claims_obj.strip()]

    return []


def extract_year(raw: Dict) -> Optional[int]:
    return _first_int(
        raw.get("year"),
        raw.get("publication_year"),
        raw.get("filing_year"),
        raw.get("date"),
        raw.get("publication_date"),
        (raw.get("bibliographic") or {}).get("publication_date") if isinstance(raw.get("bibliographic"), dict) else None,
    )


def extract_cpc_codes(raw: Dict) -> List[str]:
    from app.services.ingestion.load_hupd import _extract_cpc_codes
    return _extract_cpc_codes(raw)


def parse_patent(raw: Dict) -> ParsedPatent:
    pid = extract_patent_id(raw)
    title = extract_title(raw)
    abstract = extract_abstract(raw)
    claims = extract_claims(raw)
    year = extract_year(raw)
    cpcs = extract_cpc_codes(raw)

    return ParsedPatent(
        patent_id=pid,
        title=title,
        abstract=abstract,
        claims=claims,
        year=year,
        cpc_codes=cpcs,
    )
