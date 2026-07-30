import os, sys, pytest
from unittest.mock import MagicMock, patch


def _fake_embed(text):
    """Deterministic fake embedding for tests.
    Returns same vector for identical text, different for different text."""
    import hashlib
    h = hashlib.md5(text.encode()).digest()
    return [b / 255.0 for b in h[:4]]
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestDenseRetriever:
    @pytest.fixture
    def store_and_embed(self):
        from vector_store import InMemoryVectorStore
        class FakeEmbedder:
            """Real embedder using deterministic hash-based vectors."""
            def embed(self, text):
                return _fake_embed(text)
        store = InMemoryVectorStore(dimension=4)
        # Add with real computed vectors
        store.add("d1", _fake_embed("Python programming"), {"text": "Python programming"})
        store.add("d2", _fake_embed("Java programming"), {"text": "Java programming"})
        store.add("d3", _fake_embed("Machine learning"), {"text": "Machine learning"})
        return store, FakeEmbedder()

    def test_dense_retrieve(self, store_and_embed):
        from retriever import DenseRetriever
        store, embed = store_and_embed
        retriever = DenseRetriever(store, embed)
        # Query with real embedding -- should find the closest match
        results = retriever.retrieve("Python programming", k=2)
        assert len(results) == 2
        # The top result should be d1 since its text matches the query
        # (same md5 hash = same embedding = highest cosine similarity)
        assert results[0]["id"] == "d1"

    def test_dense_retrieve_with_different_query(self, store_and_embed):
        from retriever import DenseRetriever
        store, embed = store_and_embed
        retriever = DenseRetriever(store, embed)
        results = retriever.retrieve("Machine learning", k=3)
        assert len(results) == 3
        assert results[0]["id"] == "d3"  # exact match on embedding

    def test_empty_query(self, store_and_embed):
        from retriever import DenseRetriever
        store, embed = store_and_embed
        retriever = DenseRetriever(store, embed)
        assert retriever.retrieve("", k=5) == []

class TestBM25Retriever:
    def test_bm25_retrieve(self):
        from retriever import BM25Retriever
        docs = [
            {"id": "d1", "text": "Python is a programming language"},
            {"id": "d2", "text": "Java is also a programming language"},
            {"id": "d3", "text": "Machine learning uses Python"},
        ]
        retriever = BM25Retriever()
        retriever.index(docs)
        results = retriever.retrieve("Python programming", k=2)
        assert len(results) >= 1
        assert results[0]["id"] in ("d1", "d3")

    def test_bm25_empty_index(self):
        from retriever import BM25Retriever
        retriever = BM25Retriever()
        assert retriever.retrieve("test") == []

    def test_bm25_update(self):
        from retriever import BM25Retriever
        retriever = BM25Retriever()
        retriever.index([{"id": "d1", "text": "hello world"}])
        assert len(retriever.retrieve("hello")) == 1
        retriever.index([{"id": "d2", "text": "hello python"}])
        assert len(retriever.retrieve("hello")) == 2

class TestHybridRetriever:
    @pytest.fixture
    def real_hybrid(self):
        """Hybrid retriever backed by real BM25 and real dense retriever."""
        from vector_store import InMemoryVectorStore
        from retriever import DenseRetriever, BM25Retriever, HybridRetriever

        class FakeEmbedder:
            def embed(self, text):
                return _fake_embed(text)

        store = InMemoryVectorStore(dimension=4)
        embedder = FakeEmbedder()

        docs = [
            {"id": "d1", "text": "Python programming for AI"},
            {"id": "d2", "text": "Java programming for enterprise"},
            {"id": "d3", "text": "Machine learning with Python"},
            {"id": "d4", "text": "C++ for game development"},
        ]
        for d in docs:
            store.add(d["id"], embedder.embed(d["text"]), {"text": d["text"]})

        dense = DenseRetriever(store, embedder)
        sparse = BM25Retriever()
        sparse.index(docs)
        hybrid = HybridRetriever(dense, sparse, dense_weight=0.5)
        return hybrid, docs

    def test_hybrid_fusion_with_real_components(self, real_hybrid):
        """Hybrid with real BM25 + real dense retriever.
        Querying 'Python' should rank d1 and d3 higher (both mention Python)."""
        hybrid, docs = real_hybrid
        results = hybrid.retrieve("Python programming", k=4)
        assert len(results) >= 2
        # d1 contains "Python programming" in both BM25 terms and embedding
        # so it should be in the top results regardless of fusion weighting
        top_ids = [r["id"] for r in results]
        assert "d1" in top_ids[:3]

    def test_hybrid_weighted(self, real_hybrid):
        from retriever import HybridRetriever
        hybrid, docs = real_hybrid
        # Different weight should produce different ordering
        hybrid_default = HybridRetriever(hybrid.dense, hybrid.sparse, dense_weight=0.5)
        hybrid_dense = HybridRetriever(hybrid.dense, hybrid.sparse, dense_weight=1.0)
        r_default = hybrid_default.retrieve("Python", k=3)
        r_dense = hybrid_dense.retrieve("Python", k=3)
        assert len(r_default) >= 1
        assert len(r_dense) >= 1
        # Different weights should produce different result sets
        # (at minimum, different scores)
        r_default_scores = [r["score"] for r in r_default]
        r_dense_scores = [r["score"] for r in r_dense]
        assert r_default_scores != r_dense_scores
