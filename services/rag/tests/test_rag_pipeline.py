import os, sys, pytest
from unittest.mock import MagicMock, patch
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestRAGPipeline:
    @pytest.fixture
    def pipeline(self):
        from rag_pipeline import RAGPipeline
        from config import RAGConfig
        p = RAGPipeline(config=RAGConfig(embedding_model="test", chunk_size=100))
        # Mock components
        p._embedding_model = MagicMock()
        p._embedding_model.embed.return_value = [0.1]*4
        p._vector_store = MagicMock()
        p._vector_store.search.return_value = [
            {"id": "d1", "score": 0.95, "metadata": {"text": "Python is a programming language."}},
            {"id": "d2", "score": 0.80, "metadata": {"text": "Python is used for AI."}},
        ]
        p._llm_engine = MagicMock()
        p._llm_engine.generate.return_value.chunk = "Python is a programming language."
        return p

    def test_ingest(self, pipeline):
        pipeline.ingest("Python is great for machine learning and AI.")
        # embed_batch is called for multiple chunks
        assert pipeline._embedding_model.embed_batch.called or pipeline._embedding_model.embed.called
        pipeline._vector_store.add_batch.assert_called_once()

    def test_ingest_batch(self):
        from rag_pipeline import RAGPipeline
        p = RAGPipeline()
        p._embedding_model = MagicMock()
        p._embedding_model.embed_batch.return_value = [[0.1]*4]
        ids = p.ingest_batch(["doc1", "doc2"])
        assert len(ids) == 2

    def test_query(self, pipeline):
        result = pipeline.query("What is Python?")
        assert "answer" in result or "context" in result

    def test_query_with_sources(self, pipeline):
        result = pipeline.query("Tell me about Python")
        assert "sources" in result or "context" in result

    def test_query_empty(self, pipeline):
        pipeline._vector_store.search.return_value = []
        result = pipeline.query("Unknown topic")
        assert result is not None

    def test_build_prompt(self, pipeline):
        contexts = ["Python is a language.", "Python is used for AI."]
        prompt = pipeline.build_prompt("What is Python?", contexts)
        assert "Python is a language" in prompt
        assert "What is Python?" in prompt

    def test_build_prompt_no_context(self, pipeline):
        prompt = pipeline.build_prompt("Hello?", [])
        assert "Hello?" in prompt

    def test_clear_index(self, pipeline):
        pipeline.clear()
        pipeline._vector_store.clear.assert_called_once()

    def test_document_count(self, pipeline):
        pipeline._vector_store.count.return_value = 5
        assert pipeline.document_count() == 5

    def test_format_context_no_truncation(self, pipeline):
        contexts = [{"metadata": {"text": "Short doc."}}]
        formatted = pipeline.format_context(contexts, max_tokens=1000)
        assert "Short doc." in formatted

    def test_format_context_truncation(self, pipeline):
        contexts = [{"metadata": {"text": "Long " * 500}}]
        formatted = pipeline.format_context(contexts, max_tokens=100)
        assert len(formatted) < len("Long " * 500)

class TestRAGWithStreaming:
    @pytest.fixture
    def pipeline(self):
        from rag_pipeline import RAGPipeline
        p = RAGPipeline()
        p._embedding_model = MagicMock()
        p._embedding_model.embed.return_value = [0.1]*4
        p._vector_store = MagicMock()
        p._vector_store.search.return_value = [
            {"id": "d1", "score": 0.9, "metadata": {"text": "Relevant context."}}
        ]
        mock_llm = MagicMock()
        mock_llm.stream_generate.return_value = iter([
            type('R', (), {'chunk': 'Hello', 'is_finished': False, 'generated_tokens': 1})(),
            type('R', (), {'chunk': ' world', 'is_finished': False, 'generated_tokens': 2})(),
            type('R', (), {'chunk': '', 'is_finished': True, 'generated_tokens': 2})(),
        ])
        p._llm_engine = mock_llm
        return p

    def test_stream_query(self, pipeline):
        chunks = list(pipeline.stream_query("Hello"))
        assert len(chunks) >= 2
        found_final = any(getattr(c, 'is_finished', False) or (isinstance(c, dict) and c.get('is_finished')) for c in chunks)
        assert found_final
