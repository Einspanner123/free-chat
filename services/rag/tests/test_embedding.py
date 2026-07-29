import os, sys, pytest
from unittest.mock import MagicMock, patch
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestEmbeddingModel:
    def test_init(self):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test-model", dimension=384)
        assert model.model_name == "test-model"
        assert model.dimension == 384

    @patch('embedding.EmbeddingModel._load_model')
    def test_embed_single(self, mock_load):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test")
        model._model = MagicMock()
        model._model.encode.return_value = [[0.1] * 384]
        vec = model.embed("Hello world")
        assert len(vec) == 384
        assert abs(vec[0] - 0.1) < 1e-6

    @patch('embedding.EmbeddingModel._load_model')
    def test_embed_batch(self, mock_load):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test")
        model._model = MagicMock()
        model._model.encode.return_value = [[0.1]*384, [0.2]*384]
        vecs = model.embed_batch(["Hello", "World"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

    @patch('embedding.EmbeddingModel._load_model')
    def test_embed_empty(self, mock_load):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test")
        assert model.embed("") == []
        assert model.embed_batch([]) == []

    @patch('embedding.EmbeddingModel._load_model')
    def test_similarity(self, mock_load):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test")
        model._model = MagicMock()
        model._model.encode.side_effect = [[[1.0, 0.0]], [[1.0, 0.0]]]
        sim = model.similarity("cat", "cat")
        assert abs(sim - 1.0) < 0.01

    @patch('embedding.EmbeddingModel._load_model')
    def test_similarity_different(self, mock_load):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test")
        model._model = MagicMock()
        model._model.encode.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]
        sim = model.similarity("cat", "dog")
        assert abs(sim) < 0.5

class TestEmbeddingCache:
    @patch('embedding.EmbeddingModel._load_model')
    def test_cache_hit(self, mock_load):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test")
        model._model = MagicMock()
        model._model.encode.return_value = [[0.5]*384]
        v1 = model.embed("Hello")
        v2 = model.embed("Hello")  # should hit cache
        assert len(v1) == len(v2)
        assert model._model.encode.call_count == 1  # only called once

    @patch('embedding.EmbeddingModel._load_model')
    def test_cache_clear(self, mock_load):
        from embedding import EmbeddingModel
        model = EmbeddingModel(model_name="test")
        model._model = MagicMock()
        model._model.encode.return_value = [[0.5]*384]
        model.embed("Hello")
        model.embed("Hello")
        model.clear_cache()
        model.embed("Hello")  # should re-compute
        assert model._model.encode.call_count == 2  # cache cleared, so 2nd call
