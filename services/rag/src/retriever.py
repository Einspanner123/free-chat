"""Retrieval strategies: dense, sparse (BM25), and hybrid fusion."""

import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter


class DenseRetriever:
    """Dense retrieval using embedding similarity."""

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents by dense embedding similarity.

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            List of result dicts with id, score, metadata.
        """
        if not query:
            return []
        query_vec = self.embedding_model.embed(query)
        if not query_vec:
            return []
        return self.vector_store.search(query_vec, k=k)


class BM25Retriever:
    """Sparse retrieval using BM25 (Okapi BM25 variant)."""

    def __init__(self):
        self._documents: List[Dict] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_len: float = 0.0
        self._total_docs: int = 0
        self._k1: float = 1.5
        self._b: float = 0.75

    def index(self, documents: List[Dict[str, str]]):
        """Index documents for BM25 retrieval.

        Args:
            documents: List of dicts with "id" and "text" keys.
        """
        for doc in documents:
            doc_id = doc.get("id", str(len(self._documents)))
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            self._documents.append({"id": doc_id, "text": text, "tokens": tokens})
            self._total_docs += 1
            self._avg_doc_len = (
                (self._avg_doc_len * (self._total_docs - 1) + len(tokens))
                / self._total_docs
            )
            # Update document frequency
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freq[term] += 1

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents by BM25 score.

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            List of result dicts with id, score, text.
        """
        if not self._documents or not query:
            return []
        query_tokens = self._tokenize(query)
        scores = []

        for doc in self._documents:
            score = self._bm25_score(query_tokens, doc["tokens"])
            if score > 0:
                scores.append({
                    "id": doc["id"],
                    "score": score,
                    "text": doc["text"],
                })

        scores.sort(key=lambda x: -x["score"])
        return scores[:k]

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute BM25 score for a document."""
        doc_len = len(doc_tokens)
        doc_counter = Counter(doc_tokens)
        score = 0.0

        for term in set(query_tokens):
            tf = doc_counter.get(term, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(term, 1)
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = tf * (self._k1 + 1) / (tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_doc_len))
            score += idf * tf_norm

        return score

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer: lowercase and split on non-alphanumeric."""
        return re.findall(r'\w+', text.lower())


class HybridRetriever:
    """Hybrid retrieval fusing dense and sparse scores."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        dense_weight: float = 0.5,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using hybrid dense-sparse fusion.

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            List of result dicts with id, score (fused), text.
        """
        if not query:
            return []

        dense_results = self.dense.retrieve(query, k=k * 2)
        sparse_results = self.sparse.retrieve(query, k=k * 2)

        # Normalize scores to [0, 1]
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)

        # Fuse scores
        fused: Dict[str, Dict] = {}
        for r in dense_results:
            fused[r["id"]] = {
                "id": r["id"],
                "score": self.dense_weight * r.get("norm_score", r["score"]),
                "text": r.get("text", r.get("metadata", {}).get("text", "")),
                "metadata": r.get("metadata", {}),
            }

        for r in sparse_results:
            if r["id"] in fused:
                fused[r["id"]]["score"] += (1 - self.dense_weight) * r.get("norm_score", r["score"])
            else:
                fused[r["id"]] = {
                    "id": r["id"],
                    "score": (1 - self.dense_weight) * r.get("norm_score", r["score"]),
                    "text": r.get("text", ""),
                }

        results = sorted(fused.values(), key=lambda x: -x["score"])
        return results[:k]

    @staticmethod
    def _normalize_scores(results: List[Dict]) -> List[Dict]:
        """Normalize scores to [0, 1] using min-max."""
        if not results:
            return results
        scores = [r["score"] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s == 0:
            for r in results:
                r["norm_score"] = 1.0
        else:
            for r in results:
                r["norm_score"] = (r["score"] - min_s) / (max_s - min_s)
        return results
