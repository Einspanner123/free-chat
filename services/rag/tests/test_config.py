import os, sys, pytest
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestRAGConfig:
    def test_defaults(self):
        from config import RAGConfig
        cfg = RAGConfig()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.top_k == 5
        assert cfg.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert cfg.retrieval_strategy == "hybrid"
    def test_custom(self):
        from config import RAGConfig
        cfg = RAGConfig(chunk_size=256, top_k=10, retrieval_strategy="dense")
        assert cfg.chunk_size == 256
        assert cfg.top_k == 10
    def test_strategy_validation(self):
        from config import RAGConfig
        for s in ["dense", "sparse", "hybrid"]: RAGConfig(retrieval_strategy=s)
        with pytest.raises(ValueError): RAGConfig(retrieval_strategy="invalid")

class TestChunkerConfig:
    def test_defaults(self):
        from config import ChunkerConfig
        cfg = ChunkerConfig()
        assert cfg.strategy == "recursive"
        assert cfg.separators == ["\n\n", "\n", ".", "!", "?", " ", ""]
    def test_custom_separators(self):
        from config import ChunkerConfig
        cfg = ChunkerConfig(separators=["\n\n", "\n"])
        assert len(cfg.separators) == 2
