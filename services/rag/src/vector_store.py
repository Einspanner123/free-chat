"""Vector store abstraction with in-memory and ChromaDB implementations."""

import json
import math
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from loguru import logger


class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict] = {}

    def add(self, id: str, vector: List[float], metadata: Optional[Dict] = None):
        """Add a single vector."""
        self._vectors[id] = vector
        self._metadata[id] = metadata or {}

    def add_batch(self, ids: List[str], vectors: List[List[float]], metadatas: Optional[List[Dict]] = None):
        """Add multiple vectors."""
        if metadatas is None:
            metadatas = [{} for _ in ids]
        for id, vec, meta in zip(ids, vectors, metadatas):
            self.add(id, vec, meta)

    def search(self, query_vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Search for nearest neighbors by cosine similarity.

        Args:
            query_vector: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of {id, score, metadata} dicts sorted by descending score.
        """
        if not self._vectors:
            return []

        scores = []
        for vid, vec in self._vectors.items():
            score = self._cosine_similarity(query_vector, vec)
            scores.append((vid, score))

        scores.sort(key=lambda x: -x[1])
        results = []
        for vid, score in scores[:k]:
            results.append({
                "id": vid,
                "score": score,
                "metadata": self._metadata.get(vid, {}),
            })
        return results

    def delete(self, id: str):
        """Delete a vector by ID."""
        self._vectors.pop(id, None)
        self._metadata.pop(id, None)

    def clear(self):
        """Clear all vectors."""
        self._vectors.clear()
        self._metadata.clear()

    def count(self) -> int:
        return len(self._vectors)

    def save(self, path: str):
        """Save index to a JSON file."""
        data = {
            "dimension": self.dimension,
            "vectors": {k: v for k, v in self._vectors.items()},
            "metadata": {k: v for k, v in self._metadata.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        """Load index from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.dimension = data.get("dimension", self.dimension)
        self._vectors = data.get("vectors", {})
        self._metadata = data.get("metadata", {})

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class ChromaDBStore:
    """ChromaDB-based vector store (production)."""

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_docs"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init()

    def _init(self):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            logger.info(f"ChromaDB initialized: {self.collection_name}")
        except ImportError:
            logger.warning("chromadb not installed")

    def close(self):
        self._client = None
        self._collection = None
