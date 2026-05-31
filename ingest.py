#!/usr/bin/env python3
"""
ingest.py — Ingestion pipeline for the Personal Knowledge Assistant.

Orchestrates the full pipeline:
  1. Parse  — extract text from PDFs / images (OCR fallback for scanned pages)
  2. Chunk  — split pages into overlapping TextChunks
  3. Embed  — encode with sentence-transformers on local GPU/CPU
  4. Store  — upsert into persistent ChromaDB collection
  5. Update — write metadata.json for the Streamlit sidebar

Usage examples:
  python ingest.py                          # ingest data/examples/ (demo docs)
  python ingest.py --dir data/raw           # ingest personal docs
  python ingest.py --dir data/raw --reset   # wipe DB, then re-ingest everything
  python ingest.py --dir data/raw --verbose # detailed debug output
  python ingest.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Configure logging before any project imports so we catch import-time messages.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

from src.ingestion.parser import parse_directory
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import (
    Embedder,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MODEL_NAME,
)
from src.utils.metadata import update as update_metadata


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest documents into the Personal Knowledge Assistant vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dir",
        default="data/examples",
        metavar="PATH",
        help="Directory containing documents to ingest.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection before ingesting (full re-index).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        metavar="N",
        help="Target max characters per chunk.",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        metavar="N",
        help="Characters of context carried from each chunk into the next.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        metavar="MODEL",
        help="sentence-transformers model identifier.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help="Chunks per encode + upsert batch.",
    )
    p.add_argument(
        "--chroma-path",
        default=DEFAULT_CHROMA_PATH,
        metavar="PATH",
        help="Directory where ChromaDB persists data.",
    )
    p.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        metavar="NAME",
        help="ChromaDB collection name.",
    )
    p.add_argument(
        "--metadata-path",
        default="metadata.json",
        metavar="PATH",
        help="Path to metadata.json (read by the Streamlit sidebar).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _divider(label: str = "") -> None:
    line = "─" * 60
    logger.info(line)
    if label:
        logger.info(label)


def run(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    doc_dir = Path(args.dir)
    if not doc_dir.exists():
        logger.error("Directory not found: %s", doc_dir)
        return 1

    # ── Step 1 — Parse ────────────────────────────────────────────────────────
    _divider("STEP 1/4 — Parsing documents in: %s" % doc_dir)

    pages = parse_directory(doc_dir)
    if not pages:
        logger.error("No supported documents found in %s. Nothing to ingest.", doc_dir)
        return 1

    num_files = len({p.source_file for p in pages})
    logger.info("Parsed %d page(s) from %d file(s).", len(pages), num_files)

    # ── Step 2 — Chunk ────────────────────────────────────────────────────────
    _divider("STEP 2/4 — Chunking  [size=%d  overlap=%d]" % (args.chunk_size, args.chunk_overlap))

    chunks = chunk_pages(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        logger.error("Chunking produced 0 chunks — check document content.")
        return 1

    logger.info("Produced %d chunk(s) from %d page(s).", len(chunks), len(pages))

    # ── Step 3 — Embed + store ────────────────────────────────────────────────
    _divider("STEP 3/4 — Embedding + storing in ChromaDB")

    embedder = Embedder(
        model_name=args.model,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
    )

    if args.reset:
        embedder.reset()

    stored = embedder.embed_and_store(chunks, batch_size=args.batch_size)
    total_in_db = embedder.count()
    logger.info(
        "Stored %d chunk(s) this run — %d total in collection '%s'.",
        stored,
        total_in_db,
        args.collection,
    )

    # ── Step 4 — Update metadata ──────────────────────────────────────────────
    _divider("STEP 4/4 — Updating %s" % args.metadata_path)

    meta = update_metadata(
        chunks=chunks,
        total_chunks_in_db=total_in_db,
        path=args.metadata_path,
        reset=args.reset,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    _divider()
    logger.info("INGESTION COMPLETE")
    logger.info("  Documents  : %d", meta["num_docs"])
    logger.info("  Chunks     : %d", meta["num_chunks"])
    logger.info("  Updated at : %s", meta["last_updated"])
    logger.info("  ChromaDB   : %s", args.chroma_path)
    _divider()

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = _build_parser()
    sys.exit(run(parser.parse_args()))
