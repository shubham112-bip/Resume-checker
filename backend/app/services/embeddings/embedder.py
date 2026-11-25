"""Local transformer embedding service using sentence-transformers (BAAI/bge-small-en-v1.5)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from app.config import get_settings

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Embed search queries with appropriate query prefix."""

    @abstractmethod
    def embed_passages(self, passages: Sequence[str]) -> np.ndarray:
        """Embed text passages/chunks without query prefix."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or get_settings().embedding_model
        logger.info("Loading embedding model %s...", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        # BGE models use 384 dimensions
        self._dim = self.model.get_embedding_dimension()

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, self._dim), dtype=np.float32)
        prefixed = [
            f"{BGE_QUERY_PREFIX}{q}" if "bge" in self.model_name.lower() else q
            for q in queries
        ]
        return self.model.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def embed_passages(self, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.empty((0, self._dim), dtype=np.float32)
        return self.model.encode(
            list(passages), normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class HashingEmbedder(BaseEmbedder):
    """Deterministic hash-based offline fallback embedder (uses MurmurHash / feature hasher)."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        from sklearn.feature_extraction.text import HashingVectorizer

        self.hasher = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, self._dim), dtype=np.float32)
        return self.hasher.transform(queries).toarray().astype(np.float32)

    def embed_passages(self, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.empty((0, self._dim), dtype=np.float32)
        return self.hasher.transform(passages).toarray().astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


_embedder_instance: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        cfg = get_settings()
        if cfg.embedding_backend == "hashing":
            _embedder_instance = HashingEmbedder()
        else:
            try:
                _embedder_instance = SentenceTransformerEmbedder(cfg.embedding_model)
            except Exception as e:
                logger.warning(
                    "Failed to load SentenceTransformer '%s': %s. Falling back to HashingEmbedder.",
                    cfg.embedding_model,
                    e,
                )
                _embedder_instance = HashingEmbedder()
    return _embedder_instance
