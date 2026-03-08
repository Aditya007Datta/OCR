"""
Embedding store using SentenceTransformers + FAISS for local vector search.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

INDEX_DIR = Path("indexes")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "metadata.json"


class EmbeddingStore:
    """
    Local FAISS-based vector store using SentenceTransformers embeddings.
    Supports add, search, and persist/load operations.
    """

    def __init__(self, index_dir: Path = INDEX_DIR, model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None
        self._index = None
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension

        self._load_existing()

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_index(self):
        """Lazy-initialize or return existing FAISS index."""
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
            logger.info(f"Created new FAISS IndexFlatIP with dimension {self.dimension}")
        return self._index

    def _load_existing(self):
        """Load existing index and metadata from disk."""
        if INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(INDEX_FILE))
                with open(METADATA_FILE) as f:
                    data = json.load(f)
                    self._texts = data.get("texts", [])
                    self._metadata = data.get("metadata", [])
                logger.info(f"Loaded existing FAISS index with {self._index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")

    def _save(self):
        """Persist index and metadata to disk."""
        try:
            import faiss
            faiss.write_index(self._index, str(INDEX_FILE))
            with open(METADATA_FILE, "w") as f:
                json.dump({"texts": self._texts, "metadata": self._metadata}, f)
            logger.info(f"Saved FAISS index with {self._index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        model = self._get_model()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype("float32")

    def add_documents(self, texts: list[str], metadata: Optional[list[dict]] = None):
        """Add documents to the vector store."""
        if not texts:
            return

        metadata = metadata or [{} for _ in texts]
        logger.info(f"Adding {len(texts)} documents to vector store")

        # Chunk long texts
        chunked_texts, chunked_meta = self._chunk_texts(texts, metadata)

        embeddings = self._embed(chunked_texts)
        index = self._get_index()
        index.add(embeddings)

        self._texts.extend(chunked_texts)
        self._metadata.extend(chunked_meta)
        self._save()

    def _chunk_texts(
        self,
        texts: list[str],
        metadata: list[dict],
        max_chars: int = 800,
        overlap: int = 100
    ) -> tuple[list[str], list[dict]]:
        """Split long texts into overlapping chunks."""
        out_texts = []
        out_meta = []
        for text, meta in zip(texts, metadata):
            if len(text) <= max_chars:
                out_texts.append(text)
                out_meta.append(meta)
            else:
                # Split into chunks
                words = text.split()
                chunk_words = max_chars // 6
                overlap_words = overlap // 6
                for i in range(0, len(words), chunk_words - overlap_words):
                    chunk = " ".join(words[i: i + chunk_words])
                    if chunk.strip():
                        out_texts.append(chunk)
                        out_meta.append(meta)
        return out_texts, out_meta

    def search(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        """Search for the k most relevant documents."""
        if not self._texts or self._index is None or self._index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        query_embedding = self._embed([query])
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._texts):
                continue
            results.append({
                "text": self._texts[idx],
                "metadata": self._metadata[idx] if idx < len(self._metadata) else {},
                "score": float(score),
            })

        return results

    def clear(self):
        """Clear the vector store."""
        import faiss
        self._index = faiss.IndexFlatIP(self.dimension)
        self._texts = []
        self._metadata = []
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
        logger.info("Vector store cleared")

    @property
    def document_count(self) -> int:
        """Number of documents in the store."""
        return len(self._texts)
