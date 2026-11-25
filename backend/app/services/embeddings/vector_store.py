"""Vector store abstraction with local FAISS IndexFlatIP implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    section: str
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    chunk: DocumentChunk
    similarity: float  # cosine similarity between -1 and 1


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks and their normalized embedding vectors."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        """Search top-k most similar chunks."""

    @abstractmethod
    def count(self) -> int:
        """Total chunks in store."""


class FAISSVectorStore(VectorStore):
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        # Since embeddings are L2 normalized, IndexFlatIP computes exact cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunks ({len(chunks)})"
            )
        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        if self.count() == 0:
            return []
        q = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.count())
        distances, indices = self.index.search(q, k)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append(SearchResult(chunk=self.chunks[idx], similarity=float(dist)))
        return results

    def count(self) -> int:
        return self.index.ntotal
