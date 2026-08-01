"""
Cross-encoder reranking for Stage-1 patent candidates (RET-07).

Why this exists
---------------
Measured against the 6,000-patent HUPD corpus (``EVAL_ABLATION.md`` Test 5,
19 queries / 105 relevance judgments), dense retrieval places **94%** of
relevant patents inside the top 200 but only 49% inside the top 10:

    recall@10  0.4891      top 10       51 judgments
    recall@50  0.7632      ranks 11-50  30 judgments   <-- reachable by reranking
    recall@200 0.9395      never found   6 judgments

The relevant patents are retrieved; they are ordered badly. A bi-encoder
scores query and document independently, so it cannot model term interaction.
A cross-encoder reads the pair jointly and is far better at fine-grained
ordering — at a cost that only makes sense over a short candidate list.

Everything else measured was worth far less: fusion tuning +0.026 MRR, query
routing +0.117 MRR, and *neither can reach past rank 20*.

Model
-----
``Xenova/ms-marco-MiniLM-L-6-v2`` via ``fastembed``'s ONNX runtime — already a
dependency (it powers BM25), so this adds nothing to requirements.txt. 80 MB.

Note this invalidates ``MNT-03`` in AUDIT.md, which proposed dropping the ONNX
stack on the grounds that the stemmer-based BM25 model never uses it. The
cross-encoder does.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import anyio

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_encoder: Optional[Any] = None
_encoder_lock = threading.Lock()
_encoder_model_name: Optional[str] = None


@dataclass(frozen=True)
class RerankConfig:
    """Configuration for cross-encoder reranking."""

    model_name: str = DEFAULT_MODEL
    max_document_chars: int = 1200


def _get_encoder(model_name: str) -> Any:
    """
    Return the process-wide cross-encoder, creating it on first use.

    ``fastembed`` is imported lazily so importing this module does not drag in
    the ONNX stack — the same reason ``qdrant.py`` defers its BM25 import, and
    what keeps this module unit-testable without downloading a model.

    Construction downloads model artifacts, so it is guarded by a lock:
    scoring runs on worker threads and two concurrent first calls would
    otherwise each build (and download) their own instance.
    """
    global _encoder, _encoder_model_name
    if _encoder is None or _encoder_model_name != model_name:
        with _encoder_lock:
            if _encoder is None or _encoder_model_name != model_name:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                log.info(f"[RERANK] Loading cross-encoder '{model_name}'")
                _encoder = TextCrossEncoder(model_name=model_name)
                _encoder_model_name = model_name
    return _encoder


async def warm_reranker(model_name: str = DEFAULT_MODEL) -> None:
    """
    Build the cross-encoder off the event loop.

    Call once at startup so the first query does not pay the model download on
    the loop — the same failure BUG-04 fixed for the BM25 encoder.
    """
    await anyio.to_thread.run_sync(lambda: _get_encoder(model_name))


# ----------------------------------------------------------------------
# Pure helpers (unit-tested without a model)
# ----------------------------------------------------------------------

def document_text(match: Any, max_chars: int) -> str:
    """
    Build the text a cross-encoder scores against the query.

    Reads title and abstract from the Qdrant payload. Falls back through the
    same keys the store's extractors use, so a claim-level match (``text``) is
    handled as gracefully as a patent-level one.

    Returns "" when nothing usable is present — the caller must treat such a
    candidate as unscoreable rather than feeding an empty string to the model.
    """
    md = getattr(match, "metadata", None) or {}
    parts: List[str] = []
    for field in ("title", "abstract", "text"):
        value = md.get(field)
        if value:
            parts.append(str(value))
    return " ".join(parts)[:max_chars].strip()


def apply_scores(
    matches: Sequence[Any], scores: Sequence[float], top_n: int
) -> List[Any]:
    """
    Reorder ``matches`` by ``scores`` (descending) and truncate to ``top_n``.

    Pure and total: no model, no I/O. ``scores`` must be positionally aligned
    with ``matches``; a length mismatch raises rather than silently mis-pairing
    documents with scores, which would corrupt the ranking invisibly.
    """
    if len(matches) != len(scores):
        raise ValueError(
            f"scores length ({len(scores)}) does not match "
            f"matches length ({len(matches)})"
        )
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    order = sorted(range(len(matches)), key=lambda i: scores[i], reverse=True)
    return [matches[i] for i in order[:top_n]]


class CrossEncoderReranker:
    """
    Reranks retrieval candidates with a cross-encoder.

    Failure is never fatal: :meth:`rerank` returns the original ordering
    truncated to ``top_n`` if scoring fails for any reason. Retrieval degrading
    to bi-encoder order is a quality regression; raising would be an outage.
    """

    def __init__(self, cfg: Optional[RerankConfig] = None):
        self.cfg = cfg or RerankConfig()

    @classmethod
    def from_env(cls) -> "CrossEncoderReranker":
        import os

        return cls(
            RerankConfig(
                model_name=os.getenv("RERANK_MODEL", DEFAULT_MODEL),
                max_document_chars=int(os.getenv("RERANK_MAX_DOC_CHARS", "1200")),
            )
        )

    def _score(self, query: str, documents: List[str]) -> List[float]:
        """Blocking cross-encoder scoring. Always called on a worker thread."""
        return list(_get_encoder(self.cfg.model_name).rerank(query, documents))

    async def rerank(
        self, query: str, matches: Sequence[Any], top_n: int
    ) -> List[Any]:
        """
        Return the ``top_n`` matches, reordered by cross-encoder relevance.

        Scoring is CPU-bound, so it runs via ``anyio.to_thread.run_sync`` —
        without that the event loop stalls for the whole batch, which is
        precisely the defect BUG-04 fixed for BM25 encoding.
        """
        if not query:
            raise ValueError("query is required")
        if not matches:
            return []
        if len(matches) <= 1:
            return list(matches[:top_n])

        texts, scoreable, unscoreable = [], [], []
        for m in matches:
            text = document_text(m, self.cfg.max_document_chars)
            if text:
                texts.append(text)
                scoreable.append(m)
            else:
                unscoreable.append(m)

        if not scoreable:
            log.warning(
                "[RERANK] No candidate produced usable text — keeping original order"
            )
            return list(matches[:top_n])

        try:
            scores = await anyio.to_thread.run_sync(
                lambda: self._score(query, texts)
            )
            ranked = apply_scores(scoreable, scores, top_n=len(scoreable))
        except Exception:
            log.warning(
                "[RERANK] Cross-encoder scoring failed — falling back to "
                "retrieval order",
                exc_info=True,
            )
            return list(matches[:top_n])

        if unscoreable:
            log.debug(
                f"[RERANK] {len(unscoreable)} candidate(s) had no text; "
                f"ranked below scored results"
            )
        return (ranked + unscoreable)[:top_n]
