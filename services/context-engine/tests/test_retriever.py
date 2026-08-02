"""
Tests for context-engine retriever layer.
Red phase: all fail, then implemented.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


class TestBM25Retriever:
    def test_retrieve_top1(self):
        from retriever import BM25ContextRetriever
        docs = [
            {"id": "P1", "text": "Apple makes phones in California."},
            {"id": "P2", "text": "Banana is a tropical fruit."},
            {"id": "P3", "text": "Orange juice is popular."},
        ]
        retriever = BM25ContextRetriever()
        retriever.index(docs)
        results = retriever.retrieve("Apple phone California", k=1)
        assert len(results) == 1
        assert results[0]["id"] == "P1"

    def test_retrieve_top3(self):
        from retriever import BM25ContextRetriever
        docs = [{"id": f"P{i}", "text": f"document number {i} about python"} for i in range(10)]
        retriever = BM25ContextRetriever()
        retriever.index(docs)
        results = retriever.retrieve("python document", k=3)
        assert len(results) == 3

    def test_empty_query(self):
        from retriever import BM25ContextRetriever
        retriever = BM25ContextRetriever()
        retriever.index([{"id": "P1", "text": "hello"}])
        assert retriever.retrieve("", k=1) == []

    def test_no_docs(self):
        from retriever import BM25ContextRetriever
        retriever = BM25ContextRetriever()
        assert retriever.retrieve("hello", k=1) == []

    def test_format_context(self):
        from retriever import BM25ContextRetriever
        docs = [{"id": "P5", "text": "key content here"}]
        retriever = BM25ContextRetriever()
        retriever.index(docs)
        results = retriever.retrieve("key content", k=1)
        ctx = retriever.format_results(results, docs)
        assert "P5" in ctx or "key content" in ctx


class TestKeywordRetriever:
    def test_retrieve_by_keywords(self):
        from retriever import KeywordContextRetriever
        docs = [
            {"id": "P1", "text": "Apple and iPhone."},
            {"id": "P2", "text": "Banana fruit."},
        ]
        retriever = KeywordContextRetriever()
        retriever.index(docs)
        results = retriever.retrieve("iPhone Apple", k=1)
        assert results[0]["id"] == "P1"

    def test_no_match(self):
        from retriever import KeywordContextRetriever
        docs = [{"id": "P1", "text": "banana"}]
        retriever = KeywordContextRetriever()
        retriever.index(docs)
        assert retriever.retrieve("apple", k=1) == []


class TestRetrieverFactory:
    def test_create_bm25(self):
        from retriever import RetrieverFactory
        r = RetrieverFactory.create("bm25")
        assert r is not None

    def test_create_keyword(self):
        from retriever import RetrieverFactory
        r = RetrieverFactory.create("keyword")
        assert r is not None

    def test_create_dense_optional(self):
        from retriever import RetrieverFactory
        # dense requires embedding model; skip in unit test
        try:
            r = RetrieverFactory.create("dense")
            assert r is not None
        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_invalid(self):
        from retriever import RetrieverFactory
        with pytest.raises(ValueError):
            RetrieverFactory.create("invalid")
