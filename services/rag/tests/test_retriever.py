import os, sys, pytest
from unittest.mock import MagicMock, patch
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestDenseRetriever:
    @pytest.fixture
    def store_and_embed(self):
        from vector_store import InMemoryVectorStore
        store = InMemoryVectorStore(dimension=4)
        store.add("d1", [1,0,0,0], {"text": "Python programming"})
        store.add("d2", [0,1,0,0], {"text": "Java programming"})
        store.add("d3", [0,0,1,0], {"text": "Machine learning"})

        mock_embed = MagicMock()
        mock_embed.embed.return_value = [1.0, 0.0, 0.0, 0.0]
        return store, mock_embed

    def test_dense_retrieve(self, store_and_embed):
        from retriever import DenseRetriever
        store, embed = store_and_embed
        retriever = DenseRetriever(store, embed)
        results = retriever.retrieve("Python", k=2)
        assert len(results) == 2
        assert results[0]["id"] == "d1"

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
    def test_hybrid_fusion(self):
        from retriever import HybridRetriever
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            {"id": "d1", "score": 0.9, "text": "Python AI"},
            {"id": "d2", "score": 0.7, "text": "Java"},
        ]
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [
            {"id": "d2", "score": 0.8, "text": "Java"},
            {"id": "d3", "score": 0.6, "text": "C++"},
        ]
        hybrid = HybridRetriever(dense_retriever=mock_dense, sparse_retriever=mock_sparse)
        results = hybrid.retrieve("programming", k=3)
        assert len(results) >= 2
        # d1 from dense + d2 from sparse, both have equal fused scores after normalization
        # d1 and d2 both appear in top results
        top_ids = [r["id"] for r in results]
        assert "d1" in top_ids
        assert "d2" in top_ids

    def test_hybrid_weighted(self):
        from retriever import HybridRetriever
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [{"id": "d1", "score": 0.9, "text": "AI"}]
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [{"id": "d2", "score": 0.8, "text": "AI"}]
        hybrid = HybridRetriever(dense_retriever=mock_dense, sparse_retriever=mock_sparse, dense_weight=0.7)
        results = hybrid.retrieve("AI")
        assert len(results) >= 1
