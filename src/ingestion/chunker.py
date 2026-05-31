"""
chunker.py — Structure-aware text chunking for the Personal Knowledge Assistant.

Splits ParsedPage objects into TextChunk objects suitable for embedding + ChromaDB.

Splitting strategy (hierarchical, tried left-to-right):
  \\n\\n  →  \\n  →  sentence endings (. ? ! ;)  →  spaces  →  hard char split

Each separator is tried only when the current piece still exceeds chunk_size.
The separator is kept attached to its preceding piece so the text is preserved.

Overlap: the tail of each chunk is prepended to the next chunk (character-level),
giving the LLM enough context to reconstruct meaning at chunk boundaries.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.parser import ParsedPage

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 500      # target max characters per chunk
DEFAULT_CHUNK_OVERLAP = 80    # characters of context carried into the next chunk
MIN_CHUNK_LEN = 30            # chunks shorter than this are discarded

# Tried left-to-right; "" triggers a hard character split (last resort).
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """One chunk of text ready for embedding and ChromaDB indexing."""
    text: str
    source_file: str
    page_num: int
    doc_type: str
    is_ocr: bool
    chunk_index: int        # global position across the full document list (0-based)
    chunk_id: str           # stable unique ID used as ChromaDB document ID
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Internal splitting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_keeping_sep(text: str, sep: str) -> list[str]:
    """
    Split *text* on *sep*, keeping *sep* attached to its preceding piece.

    Example:
      "Hello.\\n\\nWorld.\\n\\nFoo" split on "\\n\\n"
      → ["Hello.\\n\\n", "World.\\n\\n", "Foo"]
    """
    pattern = f"({re.escape(sep)})"
    parts = re.split(pattern, text)
    result: list[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] == sep:
            combined = parts[i] + parts[i + 1]
            if combined:
                result.append(combined)
            i += 2
        else:
            if parts[i]:
                result.append(parts[i])
            i += 1
    return result


def _merge_pieces(pieces: list[str], chunk_size: int) -> list[str]:
    """Greedily concatenate short pieces into chunks up to *chunk_size*."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """
    Recursively split *text* into pieces ≤ *chunk_size* using *separators*.

    For each separator (tried in order):
      1. Split the text.
      2. Sub-split any piece that is still too large (recurse with next separators).
      3. Merge small pieces greedily up to chunk_size.
    Falls back to a hard character split when no separator works.
    """
    if len(text) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        if sep == "":
            # Hard character split — absolute last resort.
            return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]

        pieces = _split_keeping_sep(text, sep)
        if len(pieces) <= 1:
            continue  # this separator did not split the text; try the next one

        remaining = separators[i + 1 :]

        # Recursively break any piece that is still oversized.
        fine: list[str] = []
        for piece in pieces:
            if len(piece) > chunk_size:
                fine.extend(_recursive_split(piece, chunk_size, remaining))
            else:
                fine.append(piece)

        return _merge_pieces(fine, chunk_size)

    # No separator produced a split — hard cut.
    return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Prepend the tail of chunk[i-1] to chunk[i] for all i > 0.
    The tail is taken from the *original* chunk so overlaps do not accumulate.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:].lstrip()
        result.append(tail + chunks[i] if tail else chunks[i])
    return result


def _make_chunk_id(source_file: str, page_num: int, chunk_index: int) -> str:
    """
    Build a stable, unique ID for a chunk.
    Format: {stem[:24]}_p{page}_c{index}_{md5[:8]}
    Used as ChromaDB document ID — deterministic across re-ingestions.
    """
    raw = f"{source_file}::p{page_num}::c{chunk_index}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:8]
    stem = Path(source_file).stem[:24]
    return f"{stem}_p{page_num}_c{chunk_index}_{digest}"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def chunk_page(
    page: ParsedPage,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_index_offset: int = 0,
) -> list[TextChunk]:
    """
    Split one *ParsedPage* into *TextChunk* objects.

    Args:
        page: The parsed page to split.
        chunk_size: Target max characters per chunk.
        chunk_overlap: Characters of context carried from the previous chunk.
        chunk_index_offset: Starting value for chunk_index (for global numbering).

    Returns:
        List of TextChunk objects (may be empty if page text is very short).
    """
    text = page.text.strip()
    if not text:
        return []

    raw_chunks = _recursive_split(text, chunk_size, _SEPARATORS)
    chunks_with_overlap = _apply_overlap(raw_chunks, chunk_overlap)

    result: list[TextChunk] = []
    for text_piece in chunks_with_overlap:
        text_piece = text_piece.strip()
        if len(text_piece) < MIN_CHUNK_LEN:
            logger.debug(
                "Skipping short chunk (%d chars) — page %d of %s",
                len(text_piece),
                page.page_num,
                Path(page.source_file).name,
            )
            continue

        idx = chunk_index_offset + len(result)
        result.append(TextChunk(
            text=text_piece,
            source_file=page.source_file,
            page_num=page.page_num,
            doc_type=page.doc_type,
            is_ocr=page.is_ocr,
            chunk_index=idx,
            chunk_id=_make_chunk_id(page.source_file, page.page_num, idx),
            metadata={**page.metadata},
        ))

    logger.debug(
        "Page %d of %s → %d chunk(s)",
        page.page_num,
        Path(page.source_file).name,
        len(result),
    )
    return result


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split a list of *ParsedPage* objects into *TextChunk* objects.

    Chunk indices are global across the full list (not reset per page or document).
    Call this once per ingestion run with all pages.

    Args:
        pages: Output of parse_document() or parse_directory().
        chunk_size: Target max characters per chunk.
        chunk_overlap: Characters of context carried from the previous chunk.

    Returns:
        All TextChunk objects in document/page order.
    """
    all_chunks: list[TextChunk] = []
    for page in pages:
        page_chunks = chunk_page(
            page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_index_offset=len(all_chunks),
        )
        all_chunks.extend(page_chunks)

    logger.info(
        "Chunking complete — %d page(s) → %d chunk(s)  [size=%d  overlap=%d]",
        len(pages),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
