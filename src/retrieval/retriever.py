"""
retriever.py — Semantic retrieval from ChromaDB.

Encodes a natural-language query with the same sentence-transformers model
used at index time, queries the ChromaDB collection, and returns the top-k
most relevant chunks as RetrievalResult objects.

Supports optional filtering by doc_type (single value or list) via
ChromaDB's metadata where-clause, so the agent can target a specific
document category without touching unrelated chunks.

Typical usage::

    retriever = Retriever()
    results = retriever.query("Quand expire le titre de sejour ?", k=3)

    # Filter to a single doc type
    results = retriever.query("Note en Computer Vision", k=5, doc_type="transcript")

    # Filter to several doc types
    results = retriever.query("Experiences pro", k=5, doc_type=["cv"])

    # Reuse an already-loaded Embedder (avoids double model load in app.py)
    retriever = Retriever.from_embedder(embedder)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Re-use the same defaults as the embedder so model/collection always match.
# ──────────────────────────────────────────────────────────────────────────────
from src.ingestion.embedder import (
    DEFAULT_MODEL_NAME,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    Embedder,
)

DEFAULT_TOP_K = 5

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
# Return type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """One retrieved chunk with its metadata and relevance score."""
    text: str
    source_file: str
    page_num: int
    doc_type: str
    is_ocr: bool
    chunk_index: int
    chunk_id: str
    score: float        # cosine similarity ∈ [−1, 1]; higher = more relevant

    @property
    def filename(self) -> str:
        return Path(self.source_file).name


# ──────────────────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Semantic retriever backed by ChromaDB + sentence-transformers.

    The model is loaded lazily on the first query call. Use
    ``Retriever.from_embedder(embedder)`` to reuse a model and collection
    that are already loaded (e.g. after an ingestion run in the same process).
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
        self._collection = None

        self._init_chroma()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        if chromadb is None:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Retriever connected to collection '%s' (%d chunks).",
            self.collection_name,
            self._collection.count(),
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info("Loading query encoder '%s' on %s ...", self.model_name, _DEVICE)
        self._model = SentenceTransformer(self.model_name, device=_DEVICE)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_embedder(cls, embedder: Embedder) -> "Retriever":
        """
        Build a Retriever that reuses an already-loaded Embedder's model and
        collection. Avoids a second model load when ingestion and retrieval
        run in the same process (e.g. app.py, tests).
        """
        instance = cls.__new__(cls)
        instance.model_name = embedder.model_name
        instance.chroma_path = embedder.chroma_path
        instance.collection_name = embedder.collection_name
        instance._model = embedder.get_model()   # may trigger lazy load
        instance._collection = embedder.get_collection()
        logger.debug("Retriever initialised from existing Embedder.")
        return instance

    # ── Where-clause builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_where(doc_type: str | list[str] | None) -> dict | None:
        """
        Translate a doc_type filter into a ChromaDB where-clause.

        Returns None when no filter is requested (avoids passing an empty
        dict, which ChromaDB treats as an error in some versions).
        """
        if doc_type is None:
            return None
        if isinstance(doc_type, str):
            return {"doc_type": {"$eq": doc_type}}
        if len(doc_type) == 0:
            return None
        if len(doc_type) == 1:
            return {"doc_type": {"$eq": doc_type[0]}}
        return {"doc_type": {"$in": doc_type}}

    # ── Core query ─────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        k: int = DEFAULT_TOP_K,
        doc_type: str | list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-k chunks most semantically similar to *text*.

        Args:
            text: The natural-language query.
            k: Number of results to return (capped at collection size).
            doc_type: Optional filter — a single doc_type string such as
                ``"cv"`` or ``"transcript"``, a list of accepted types, or
                ``None`` to search the whole collection.

        Returns:
            List of RetrievalResult sorted by descending cosine similarity.
            Empty list if the collection is empty or no chunks match the filter.
        """
        if not text or not text.strip():
            logger.warning("query() called with empty text — returning [].")
            return []

        total = self._collection.count()
        if total == 0:
            logger.warning("Collection '%s' is empty.", self.collection_name)
            return []

        # Cap k so ChromaDB doesn't raise when k > collection size.
        effective_k = min(k, total)

        self._ensure_model()

        # Encode query with the same normalisation as the indexed embeddings.
        query_vec = self._model.encode(
            text.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        where = self._build_where(doc_type)

        try:
            raw = self._collection.query(
                query_embeddings=[query_vec],
                n_results=effective_k,
                include=["documents", "metadatas", "distances"],
                **({"where": where} if where else {}),
            )
        except Exception as exc:
            # ChromaDB raises if the where-clause matches 0 docs.
            logger.warning("ChromaDB query failed (%s) — returning [].", exc)
            return []

        results: list[RetrievalResult] = []
        ids       = raw["ids"][0]
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        for chunk_id, text_chunk, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine space returns distance = 1 − cosine_similarity,
            # so similarity = 1 − distance.
            score = round(1.0 - dist, 4)
            results.append(RetrievalResult(
                text=text_chunk,
                source_file=meta.get("source_file", ""),
                page_num=int(meta.get("page_num", 0)),
                doc_type=meta.get("doc_type", "unknown"),
                is_ocr=bool(meta.get("is_ocr", False)),
                chunk_index=int(meta.get("chunk_index", -1)),
                chunk_id=chunk_id,
                score=score,
            ))

        logger.debug(
            "query=%r  doc_type=%r  k=%d  got=%d  top_score=%.3f",
            text[:60],
            doc_type,
            effective_k,
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ── Convenience ────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()
