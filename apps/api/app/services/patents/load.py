from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Union


def _open_text(path: Union[str, Path]):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "rt", encoding="utf-8", errors="replace")


def iter_hupd_jsonl(
    path: Union[str, Path],
) -> Iterator[Dict]:
    """
    Iterate HUPD JSONL / JSONL.GZ, yielding one dict per record.
    Skips empty/broken lines instead of crashing (robust for large corpora).
    """
    with _open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            line = (line or "").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_cpc_codes(raw: Dict) -> List[str]:
    """
    HUPD-like records can store CPC in multiple ways depending on version.
    This tries several common shapes:
      - raw["cpc_current"] = [{"section":"G","class":"06","subclass":"N", ...}, ...]
      - raw["cpcs"] / raw["cpc"] / raw["cpc_codes"] = ["G06N...", ...]
    Returns list of strings like "G06N" or "G06N7/00".
    """
    out: List[str] = []

    for key in ("cpc", "cpcs", "cpc_codes", "cpc_current_codes"):
        v = raw.get(key)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())

    v = raw.get("cpc_current")
    if isinstance(v, list):
        for item in v:
            if not isinstance(item, dict):
                continue
            section = str(item.get("section") or "").strip()
            cls = str(item.get("class") or "").strip()
            subclass = str(item.get("subclass") or "").strip()

            if section and cls and subclass:
                out.append(f"{section}{cls}{subclass}")
            else:
                code = item.get("cpc") or item.get("code") or item.get("symbol")
                if isinstance(code, str) and code.strip():
                    out.append(code.strip())

    deduped: List[str] = []
    seen = set()
    for c in out:
        c = c.strip()
        if not c:
            continue
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def filter_by_cpc_prefix(
    raw: Dict,
    allowed_prefixes: List[str],
) -> bool:
    """
    Keep patents whose CPC codes include any prefix like "G06N".
    """
    if not allowed_prefixes:
        return True
    codes = _extract_cpc_codes(raw)
    for code in codes:
        for prefix in allowed_prefixes:
            if code.startswith(prefix):
                return True
    return False


def load_hupd(
    paths: Union[str, Path, List[Union[str, Path]]],
    *,
    cpc_prefixes: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Stream patents from one or many JSONL(.gz) files, optionally filtering by CPC prefixes.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    count = 0
    for p in paths:
        for raw in iter_hupd_jsonl(p):
            if cpc_prefixes and not filter_by_cpc_prefix(raw, cpc_prefixes):
                continue
            yield raw
            count += 1
            if limit is not None and count >= limit:
                return
