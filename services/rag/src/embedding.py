"""Embedding model wrapper with caching."""

import hashlib
import math
from typing import List, Optional, Dict
from functools import lru_cache

from loguru import logger


class EmbeddingModel:
    """Text embedding model with caching."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._cache: Dict[str, List[float]] = {}

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model '{self.model_name}' loaded")
            except ImportError:
                logger.warning("sentence-transformers not installed; using fallback embeddings")
                self._model = None

    def embed(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: Input text.

        Returns:
            Embedding vector as list of floats.
        """
        if not text:
            return []

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._load_model()
        if self._model is not None:
            result = self._model.encode([text])[0]
            vec = result.tolist() if hasattr(result, 'tolist') else (list(result) if isinstance(result, (list, tuple)) else [float(result)])
        else:
            vec = self._fallback_embed(text)

        self._cache[cache_key] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        self._load_model()
        if self._model is not None and len(texts) > 1:
            encoded = self._model.encode(texts)
            return [v.tolist() if hasattr(v, 'tolist') else list(v) for v in encoded]
        return [self.embed(t) for t in texts]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        if not vec_a or not vec_b:
            return 0.0
        return self._cosine_similarity(vec_a, vec_b)

    def clear_cache(self):
        self._cache.clear()

    def _fallback_embed(self, text: str) -> List[float]:
        """Fallback: hash-based pseudo-embedding when no model available."""
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        # Pad or truncate to dimension
        if len(vec) < self.dimension:
            vec = vec * (self.dimension // len(vec) + 1)
        return vec[:self.dimension]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
