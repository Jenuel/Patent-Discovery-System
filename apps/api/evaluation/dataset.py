from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Union

_REQUIRED_FIELDS = ("query_id", "query")


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    metadata_filter: Dict[str, Any] = field(default_factory=dict)
    relevant_patent_ids: FrozenSet[str] = frozenset()
    relevant_chunk_ids: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def relevant_chunk_id_set(self) -> FrozenSet[str]:
        return frozenset(self.relevant_chunk_ids)


def parse_query_cases(lines: Iterable[str]) -> List[QueryCase]:
    """Pure parser. Raises ValueError with the offending line number on
    malformed JSON or a missing required field (query_id, query)."""
    cases: List[QueryCase] = []

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON on line {line_no}: {exc}") from exc

        missing = [f for f in _REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(
                f"Line {line_no} is missing required field(s): {missing}"
            )

        cases.append(
            QueryCase(
                query_id=data["query_id"],
                query=data["query"],
                metadata_filter=data.get("metadata_filter") or {},
                relevant_patent_ids=frozenset(data.get("relevant_patent_ids") or []),
                relevant_chunk_ids=dict(data.get("relevant_chunk_ids") or {}),
                notes=data.get("notes", ""),
            )
        )

    return cases


def load_query_cases(path: Union[str, Path]) -> List[QueryCase]:
    """Reads the file at `path` and delegates to parse_query_cases."""
    with open(path, encoding="utf-8") as f:
        return parse_query_cases(f)
