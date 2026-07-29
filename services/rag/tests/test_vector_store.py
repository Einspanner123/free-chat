import os, sys, tempfile, pytest
from unittest.mock import MagicMock
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestInMemoryVectorStore:
    @pytest.fixture
    def Store(self):
        from vector_store import InMemoryVectorStore
        return InMemoryVectorStore

    def test_add_and_search(self, Store):
        store = Store(dimension=4)
        store.add("id1", [1.0, 0.0, 0.0, 0.0], {"text": "doc1"})
        store.add("id2", [0.0, 1.0, 0.0, 0.0], {"text": "doc2"})
        results = store.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["score"] > 0.9

    def test_search_top_k(self, Store):
        store = Store(dimension=2)
        for i in range(10): store.add(f"id{i}", [i/10, 1-i/10], {"text": f"doc{i}"})
        results = store.search([1.0, 0.0], k=3)
        assert len(results) == 3

    def test_delete(self, Store):
        store = Store(dimension=2)
        store.add("id1", [1.0, 0.0], {})
        store.add("id2", [0.0, 1.0], {})
        store.delete("id1")
        assert store.count() == 1

    def test_clear(self, Store):
        store = Store(dimension=2)
        store.add("id1", [1.0, 0.0], {})
        store.clear()
        assert store.count() == 0

    def test_add_batch(self, Store):
        store = Store(dimension=2)
        ids = ["a", "b", "c"]
        vecs = [[1,0], [0,1], [1,1]]
        metas = [{}, {}, {}]
        store.add_batch(ids, vecs, metas)
        assert store.count() == 3

    def test_empty_search(self, Store):
        store = Store(dimension=2)
        assert store.search([1.0, 0.0]) == []

    def test_persistence_save_load(self, Store):
        store = Store(dimension=2)
        store.add("id1", [1.0, 0.0], {"text": "hello"})
        store.add("id2", [0.0, 1.0], {"text": "world"})
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            store.save(f.name)
            store2 = Store(dimension=2)
            store2.load(f.name)
        os.unlink(f.name)
        assert store2.count() == 2
        results = store2.search([1.0, 0.0], k=1)
        assert results[0]["id"] == "id1"

class TestChromaDBStore:
    def test_init(self):
        from vector_store import ChromaDBStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaDBStore(persist_dir=tmp, collection_name="test")
            assert store.collection_name == "test"
            store.close()
