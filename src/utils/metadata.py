"""
metadata.py — Read/write metadata.json for the Personal Knowledge Assistant.

metadata.json is the source of truth for the Streamlit sidebar (last update date,
number of indexed documents, list of files). It is written by ingest.py after
every ingestion run and read by the app at startup.

Schema::

    {
      "last_updated": "2024-01-15T14:32:10.123456+00:00",  // ISO 8601 UTC
      "num_docs": 3,
      "num_chunks": 87,
      "indexed_files": [
        {
          "filename": "cv_alex_doe.pdf",
          "source_file": "data/examples/cv_alex_doe.pdf",
          "doc_type": "cv",
          "num_chunks": 24,
          "ingested_at": "2024-01-15T14:32:10.123456+00:00"
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

DEFAULT_METADATA_PATH = "metadata.json"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _empty() -> dict[str, Any]:
    return {
        "last_updated": None,
        "num_docs": 0,
        "num_chunks": 0,
        "indexed_files": [],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load(path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """
    Load metadata.json from *path*.
    Returns an empty metadata structure if the file is missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        return _empty()
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s (%s) — starting fresh.", path, exc)
        return _empty()


def save(meta: dict[str, Any], path: str | Path = DEFAULT_METADATA_PATH) -> None:
    """Write *meta* to *path* as pretty-printed UTF-8 JSON."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.debug("Wrote metadata to %s", path)


def update(
    chunks: list[TextChunk],
    total_chunks_in_db: int,
    path: str | Path = DEFAULT_METADATA_PATH,
    reset: bool = False,
) -> dict[str, Any]:
    """
    Update metadata.json after an ingestion run and return the new metadata dict.

    Behaviour:
    - ``reset=True``: replaces indexed_files entirely with this run's files.
    - ``reset=False``: merges — re-ingested files are updated in-place, files
      not touched in this run retain their existing entry.

    Args:
        chunks: All TextChunk objects produced in this ingestion run.
        total_chunks_in_db: Current total from ``embedder.count()``.
        path: Path to metadata.json.
        reset: Whether the ChromaDB collection was wiped before this run.

    Returns:
        The updated metadata dict (already written to disk).
    """
    meta = _empty() if reset else load(path)
    now = _now_iso()

    # Summarise this run by source file
    run_files: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        sf = chunk.source_file
        if sf not in run_files:
            run_files[sf] = {
                "filename": Path(sf).name,
                "source_file": sf,
                "doc_type": chunk.doc_type,
                "num_chunks": 0,
                "ingested_at": now,
            }
        run_files[sf]["num_chunks"] += 1

    # Merge into existing entries (keyed by source_file path)
    existing: dict[str, dict[str, Any]] = {
        entry["source_file"]: entry
        for entry in meta.get("indexed_files", [])
    }
    existing.update(run_files)

    meta["indexed_files"] = sorted(existing.values(), key=lambda e: e["filename"])
    meta["num_docs"] = len(meta["indexed_files"])
    meta["num_chunks"] = total_chunks_in_db
    meta["last_updated"] = now

    save(meta, path)
    logger.info(
        "metadata.json updated — %d doc(s), %d chunk(s), last_updated=%s",
        meta["num_docs"],
        meta["num_chunks"],
        now,
    )
    return meta
