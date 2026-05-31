"""
tools.py — Tool functions and routing heuristic for the LangGraph agent.

Two retrieval tools:
  search_kb      — semantic search across the full KB, with optional doc_type filter.
  extract_dates  — same search scoped to date-bearing types (titre_sejour, contract),
                   for expiry / deadline / validity questions.

The routing heuristic (decide_route) uses keyword regex to choose a tool
without spending an extra API call on routing. Priority order:
  dates keywords  >  transcript keywords  >  cv keywords  >  full search
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.retrieval.retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Route identifiers
# ──────────────────────────────────────────────────────────────────────────────

ROUTE_SEARCH_KB     = "search_kb"
ROUTE_EXTRACT_DATES = "extract_dates"

# ──────────────────────────────────────────────────────────────────────────────
# Routing keyword patterns
# ──────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(
    r"expir|validit|valide|jusqu.?\s*quand|renouvell"
    r"|date.{0,15}(expir|valid|limit|fin)"
    r"|titre.{0,10}s[eé]jour|s[eé]jour"
    r"|contrat.{0,10}(termin|fin|expir)"
    r"|deadline|[eé]ch[eé]ance",
    re.IGNORECASE,
)

_TRANSCRIPT_RE = re.compile(
    r"\bnotes?\b|mention|r[eé]sultat|mati[eè]re|\bmodule\b|\bUE\b"
    r"|examen|relev[eé]|moyenne|transcript|obtenu|valid[eé]"
    r"|\bm[12]\b|semestre|promotion|jury",
    re.IGNORECASE,
)

_CV_RE = re.compile(
    r"exp[eé]rience|travail|emploi|poste|comp[eé]tence"
    r"|formation|dipl[oô]me|entreprise|\bstage\b|CDI|CDD"
    r"|lettre.{0,10}motivation|curriculum|parcours.{0,6}pro"
    r"|skill|projet.{0,10}pro",
    re.IGNORECASE,
)


def decide_route(question: str) -> tuple[str, str | list[str] | None]:
    """
    Heuristic router: map a question to (route, doc_type_filter).

    Returns:
        route     — one of ROUTE_SEARCH_KB or ROUTE_EXTRACT_DATES.
        doc_type  — metadata filter passed to the retriever, or None for
                    full-collection search.
    """
    q = question.strip()

    if _DATE_RE.search(q):
        # extract_dates handles its own doc_type filter internally
        return ROUTE_EXTRACT_DATES, None

    if _TRANSCRIPT_RE.search(q):
        return ROUTE_SEARCH_KB, "transcript"

    if _CV_RE.search(q):
        return ROUTE_SEARCH_KB, "cv"

    return ROUTE_SEARCH_KB, None


# ──────────────────────────────────────────────────────────────────────────────
# Tool functions
# ──────────────────────────────────────────────────────────────────────────────

def search_kb(
    retriever: Retriever,
    query: str,
    k: int = 5,
    doc_type: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over the knowledge base.

    Args:
        retriever: Shared Retriever instance (model + ChromaDB already loaded).
        query: Natural-language search string.
        k: Max number of chunks to retrieve.
        doc_type: Optional metadata filter (e.g. ``"cv"``, ``"transcript"``,
                  or a list of accepted types).  ``None`` = no filter.

    Returns:
        List of chunk dicts, sorted by descending cosine similarity.
    """
    results = retriever.query(query, k=k, doc_type=doc_type)
    chunks = [_to_dict(r) for r in results]
    logger.debug("search_kb: query=%r doc_type=%r → %d chunks", query[:50], doc_type, len(chunks))
    return chunks


def extract_dates(
    retriever: Retriever,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieval scoped to date-bearing document types.

    Targets ``titre_sejour`` and ``contract`` documents — the only types
    that contain expiry dates, validity periods, and renewal deadlines.

    Args:
        retriever: Shared Retriever instance.
        query: Natural-language question about dates / validity.
        k: Max number of chunks to retrieve.

    Returns:
        List of chunk dicts from titre_sejour / contract documents only.
    """
    results = retriever.query(
        query,
        k=k,
        doc_type=["titre_sejour", "contract"],
    )
    chunks = [_to_dict(r) for r in results]
    logger.debug("extract_dates: query=%r → %d chunks", query[:50], len(chunks))
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_dict(r: RetrievalResult) -> dict[str, Any]:
    return {
        "text":        r.text,
        "source_file": r.source_file,
        "filename":    r.filename,
        "page_num":    r.page_num,
        "doc_type":    r.doc_type,
        "is_ocr":      r.is_ocr,
        "chunk_index": r.chunk_index,
        "chunk_id":    r.chunk_id,
        "score":       r.score,
    }


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Render retrieved chunks into a numbered context block for the LLM prompt.

    Each chunk gets a provenance header so the model can cite its sources.
    """
    if not chunks:
        return "(Aucun document pertinent trouve dans la base de connaissances.)"

    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = (
            f"[Source {i} | {c['filename']} p.{c['page_num']} "
            f"| type: {c['doc_type']} | pertinence: {c['score']:.2f}]"
        )
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)
