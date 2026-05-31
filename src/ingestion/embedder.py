"""
embedder.py — Embedding generation and ChromaDB indexing.

Takes TextChunk objects from chunker.py, encodes them with a local
sentence-transformers model, and upserts them into a persistent ChromaDB
collection with their metadata for downstream filtering.

Design notes:
- Model is loaded lazily on first use (avoids GPU memory at import time).
- upsert() is used instead of add() so re-ingesting is idempotent.
- Embeddings are L2-normalized; ChromaDB collection uses cosine space
  (cosine similarity = dot product on unit vectors → faster ANN search).
- ChromaDB metadata values must be primitives (str/int/float/bool);
  nested dicts from chunk.metadata are flattened and non-primitives skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "data/chroma_db"
DEFAULT_COLLECTION_NAME = "knowledge_base"
DEFAULT_BATCH_SIZE = 64

# ──────────────────────────────────────────────────────────────────────────────
# Optional heavy imports
# ──────────────────────────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_chroma_metadata(chunk: TextChunk) -> dict:
    """
    Build a ChromaDB-compatible metadata dict for *chunk*.

    ChromaDB only accepts str, int, float, and bool values — no nested objects.
    Fields from chunk.metadata are merged if they are primitive types.
    """
    meta: dict = {
        "source_file": chunk.source_file,
        "page_num": chunk.page_num,
        "doc_type": chunk.doc_type,
        "is_ocr": chunk.is_ocr,
        "chunk_index": chunk.chunk_index,
    }
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────────────────────────────────────

class Embedder:
    """
    Generates sentence embeddings and maintains a persistent ChromaDB collection.

    ChromaDB is initialised immediately (cheap); the sentence-transformers model
    is loaded lazily on the first call to embed_and_store() to avoid allocating
    GPU memory at import time.

    Usage::

        embedder = Embedder()
        embedder.embed_and_store(chunks)
        print(embedder.count(), "chunks in DB")

        # For the retriever:
        collection = embedder.get_collection()
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        chroma_path: str | Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.model_name = model_name
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name

        self._model: Optional[SentenceTransformer] = None
        self._client = None
        self._collection = None

        self._init_chroma()

    # ── ChromaDB setup ─────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready — collection '%s' at %s (%d docs)",
            self.collection_name,
            self.chroma_path,
            self._collection.count(),
        )

    # ── Model loading ──────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info(
            "Loading embedding model '%s' on %s ...", self.model_name, _DEVICE
        )
        self._model = SentenceTransformer(self.model_name, device=_DEVICE)
        dim = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded — embedding dim: %d", dim)

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_and_store(
        self,
        chunks: list[TextChunk],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Encode *chunks* and upsert them into ChromaDB.

        Re-ingesting the same chunks is idempotent: existing IDs are overwritten,
        not duplicated. Each call logs progress at INFO level.

        Args:
            chunks: Output of chunk_pages() — must be non-empty.
            batch_size: Chunks per encode + upsert call.

        Returns:
            Number of chunks processed (may be less than len(chunks) if empty).
        """
        if not chunks:
            logger.warning("embed_and_store called with an empty chunk list — nothing to do.")
            return 0

        self._ensure_model()

        stored = 0
        total = len(chunks)

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]

            embeddings = self._model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[_build_chroma_metadata(c) for c in batch],
            )

            stored += len(batch)
            logger.info(
                "Upserted %d/%d chunks — collection total: %d",
                stored,
                total,
                self._collection.count(),
            )

        return stored

    def count(self) -> int:
        """Number of chunks currently stored in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """
        Delete and recreate the collection.

        Use this before a full re-ingestion when source documents have been
        removed and stale chunks must be purged. Incremental updates (new or
        edited documents only) should use embed_and_store() directly — upsert
        handles duplicates.
        """
        logger.warning(
            "Resetting ChromaDB collection '%s' — all %d docs will be deleted.",
            self.collection_name,
            self._collection.count(),
        )
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset — 0 docs.")

    def get_collection(self):
        """Return the underlying ChromaDB collection for use by the retriever."""
        return self._collection

    def get_model(self) -> "SentenceTransformer":
        """Return the loaded SentenceTransformer model (loads it if needed)."""
        self._ensure_model()
        return self._model


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience function
# ──────────────────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks: list[TextChunk],
    chroma_path: str | Path = DEFAULT_CHROMA_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset: bool = False,
) -> Embedder:
    """
    One-shot convenience function: create an Embedder, optionally reset the
    collection, embed *chunks*, and return the Embedder for further use.

    Typical usage in ingest.py::

        from src.ingestion.embedder import embed_and_store
        embedder = embed_and_store(chunks, reset=True)
        print(embedder.count(), "chunks indexed")

    Args:
        chunks: TextChunk list from chunk_pages().
        chroma_path: Directory where ChromaDB persists its data.
        model_name: sentence-transformers model identifier.
        collection_name: ChromaDB collection name.
        batch_size: Chunks per encode + upsert call.
        reset: If True, wipe the collection before inserting.

    Returns:
        The Embedder instance (collection stays open for the retriever).
    """
    embedder = Embedder(
        model_name=model_name,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )
    if reset:
        embedder.reset()
    embedder.embed_and_store(chunks, batch_size=batch_size)
    return embedder
