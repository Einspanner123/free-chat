import os, sys, pytest
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestRecursiveChunker:
    def get_chunker(self):
        from chunker import RecursiveChunker
        return RecursiveChunker(chunk_size=50, chunk_overlap=10)

    def test_chunk_basic(self):
        c = self.get_chunker()
        chunks = c.chunk("Hello world. This is a test. " * 5)
        assert len(chunks) > 1
        assert all(len(ch) <= 50 for ch in chunks)

    def test_chunk_single(self):
        c = self.get_chunker()
        chunks = c.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_empty(self):
        c = self.get_chunker()
        assert c.chunk("") == []
        assert c.chunk(None) == []

    def test_chunk_overlap(self):
        from chunker import RecursiveChunker
        c = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = "A " * 50 + "B " * 50 + "C " * 50
        chunks = c.chunk(text, separators=[" "])
        assert len(chunks) >= 3

    def test_chunk_preserves_all_text(self):
        from chunker import RecursiveChunker
        text = "The quick brown fox jumps over the lazy dog. " * 10
        c = RecursiveChunker(chunk_size=30, chunk_overlap=5)
        chunks = c.chunk(text, separators=[" "])
        combined = "".join(chunks)
        for word in ["quick", "brown", "fox", "jumps", "lazy"]:
            assert word in combined

    def test_chunk_metadata(self):
        c = self.get_chunker()
        chunks = c.chunk_with_metadata("Hello world. Test sentence.")
        assert len(chunks) >= 1
        assert "text" in chunks[0]
        assert "index" in chunks[0]

class TestSemanticChunker:
    def test_chunk_by_sentence(self):
        from chunker import SemanticChunker
        c = SemanticChunker()
        text = "First sentence. Second sentence. Third sentence!"
        chunks = c.chunk_by_sentence(text)
        assert len(chunks) == 3

    def test_chunk_by_paragraph(self):
        from chunker import SemanticChunker
        c = SemanticChunker()
        text = "Para1\n\nPara2\n\nPara3"
        chunks = c.chunk_by_paragraph(text)
        assert len(chunks) == 3

    def test_chunk_by_topic(self):
        from chunker import SemanticChunker
        c = SemanticChunker()
        text = "# Title 1\nContent1\n\n# Title 2\nContent2"
        chunks = c.chunk_by_topic(text)
        assert len(chunks) >= 2

class TestChunkerIntegration:
    def test_chunker_factory(self):
        from chunker import ChunkerFactory
        recursive = ChunkerFactory.create("recursive", chunk_size=100)
        assert recursive is not None
        assert hasattr(recursive, "chunk")
        semantic = ChunkerFactory.create("semantic")
        assert semantic is not None

    def test_chunker_factory_invalid(self):
        from chunker import ChunkerFactory
        with pytest.raises(ValueError): ChunkerFactory.create("invalid")
