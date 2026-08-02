"""
Retrieval layer: select relevant context chunks.

Each retriever implements the same interface:
  index(docs) -> None
  retrieve(query, k) -> List[Dict]  # sorted by relevance desc
  format_results(results, docs) -> str  # assemble context
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseContextRetriever(ABC):
    """Interface for context retrievers."""

    @abstractmethod
    def index(self, docs: List[Dict[str, str]]):
        """Index documents. Each doc: {"id": str, "text": str}."""
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 1) -> List[Dict]:
        """Retrieve top-k relevant docs. Returns list of {"id", "score"}."""
        ...

    @abstractmethod
    def format_results(self, results: List[Dict], docs: List[Dict]) -> str:
        """Assemble retrieved docs into context text."""
        ...


class BM25ContextRetriever(BaseContextRetriever):
    """BM25 (Okapi) sparse retriever. Pure Python, no external deps."""

    def __init__(self):
        self._docs: List[Dict] = []
        self._doc_freq: Dict[str, int] = {}
        self._avg_len: float = 0.0
        self._k1 = 1.5
        self._b = 0.75

    def index(self, docs: List[Dict[str, str]]):
        self._docs = list(docs)
        self._doc_freq = {}
        total_len = 0
        for doc in self._docs:
            tokens = self._tokenize(doc["text"])
            doc["_tokens"] = tokens
            total_len += len(tokens)
            for term in set(tokens):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._avg_len = total_len / len(self._docs) if self._docs else 0.0

    def retrieve(self, query: str, k: int = 1) -> List[Dict]:
        if not self._docs or not query:
            return []
        query_tokens = self._tokenize(query)
        n = len(self._docs)
        scored = []
        for doc in self._docs:
            score = self._bm25(query_tokens, doc, n)
            if score > 0:
                scored.append({"id": doc["id"], "score": score})
        scored.sort(key=lambda x: -x["score"])
        return scored[:k]

    def format_results(self, results: List[Dict], docs: List[Dict]) -> str:
        id_to_doc = {d["id"]: d for d in docs}
        parts = []
        for r in results:
            d = id_to_doc.get(r["id"])
            if d:
                parts.append(d["text"])
        return "\n".join(parts)

    def _bm25(self, query_tokens: List[str], doc: Dict, n: int) -> float:
        tokens = doc["_tokens"]
        doc_len = len(tokens)
        term_counts = {}
        for t in tokens:
            term_counts[t] = term_counts.get(t, 0) + 1
        score = 0.0
        for term in set(query_tokens):
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(term, 1)
            idf = _idf(n, df)
            denom = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_len)
            score += idf * tf * (self._k1 + 1) / denom
        return score

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return re.findall(r'\w+', text.lower())


class KeywordContextRetriever(BaseContextRetriever):
    """Keyword matching retriever: score docs by query-word hits."""

    def __init__(self):
        self._docs: List[Dict] = []

    def index(self, docs: List[Dict[str, str]]):
        self._docs = list(docs)

    def retrieve(self, query: str, k: int = 1) -> List[Dict]:
        if not self._docs or not query:
            return []
        from strategies import extract_query_words
        words = extract_query_words(query)
        if not words:
            return []
        scored = []
        for doc in self._docs:
            hits = sum(1 for w in words if w.lower() in doc["text"].lower())
            if hits > 0:
                scored.append({"id": doc["id"], "score": hits})
        scored.sort(key=lambda x: -x["score"])
        return scored[:k]

    def format_results(self, results: List[Dict], docs: List[Dict]) -> str:
        id_to_doc = {d["id"]: d for d in docs}
        return "\n".join(id_to_doc[r["id"]]["text"] for r in results if r["id"] in id_to_doc)


class RetrieverFactory:
    """Create retrievers by name."""

    @staticmethod
    def create(name: str) -> BaseContextRetriever:
        if name == "bm25":
            return BM25ContextRetriever()
        if name == "keyword":
            return KeywordContextRetriever()
        if name == "dense":
            return _create_dense()
        raise ValueError(f"Unknown retriever: {name}")


def _create_dense() -> BaseContextRetriever:
    """Dense retriever using embedding (optional dependency)."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rag", "src"))
        from embedding import EmbeddingModel
        from vector_store import InMemoryVectorStore
    except ImportError as e:
        raise ImportError("sentence-transformers required for dense retriever") from e

    class DenseContextRetriever(BaseContextRetriever):
        def __init__(self):
            self._embedder = EmbeddingModel()
            self._store = None
            self._docs = []

        def index(self, docs):
            self._docs = list(docs)
            self._store = InMemoryVectorStore(dimension=384)
            ids = [d["id"] for d in docs]
            vecs = [self._embedder.embed(d["text"]) for d in docs]
            metas = [{"text": d["text"]} for d in docs]
            self._store.add_batch(ids, vecs, metas)

        def retrieve(self, query, k=1):
            if not self._docs or not query:
                return []
            qvec = self._embedder.embed(query)
            return self._store.search(qvec, k=k)

        def format_results(self, results, docs):
            id_to_doc = {d["id"]: d for d in docs}
            return "\n".join(id_to_doc[r["id"]]["text"] for r in results if r["id"] in id_to_doc)

    return DenseContextRetriever()


def _idf(n: int, df: int) -> float:
    import math
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)
