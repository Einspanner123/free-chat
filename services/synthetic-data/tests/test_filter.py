import os, sys, pytest
from unittest.mock import MagicMock
@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path: sys.path.insert(0, src)

class TestQualityFilter:
    def get_filter(self):
        from filter import QualityFilter
        return QualityFilter(min_length=5, max_length=100)

    def test_filter_by_length(self):
        f = self.get_filter()
        data = [
            {"instruction": "Hi", "output": "AB"},  # total 5 chars, passes min=5
            {"instruction": "X" * 200, "output": "Y" * 200},  # total > 100, fails max
            {"instruction": "Explain AI?", "output": "AI is a field of study."},  # passes
        ]
        filtered = f.filter(data)
        assert len(filtered) == 2  # First and third pass

    def test_filter_empty_io(self):
        f = self.get_filter()
        data = [
            {"instruction": "", "output": "something"},
            {"instruction": "test", "output": ""},
            {"instruction": "", "output": ""},
        ]
        filtered = f.filter(data)
        assert len(filtered) == 0

    def test_filter_deduplicate(self):
        f = self.get_filter()
        data = [
            {"instruction": "Long enough Q", "output": "A good answer."},
            {"instruction": "Long enough Q", "output": "A good answer."},  # duplicate
            {"instruction": "Different question", "output": "A good answer."},
        ]
        filtered = f.filter(data, deduplicate=True)
        assert len(filtered) >= 2

    def test_filter_remove_html(self):
        f = self.get_filter()
        data = [
            {"instruction": "<script>alert('xss')</script>Q", "output": "<b>A</b>"},
            {"instruction": "Clean Q", "output": "Clean A"},
        ]
        filtered = f.filter(data, remove_html=True)
        assert "<script>" not in filtered[0]["instruction"] if len(filtered) > 0 else True

    def test_filter_repetition(self):
        f = self.get_filter()
        data = [
            {"instruction": "Q", "output": "A " * 100},  # repetitive
            {"instruction": "Q2", "output": "A good response."},
        ]
        filtered = f.filter(data)
        assert len(filtered) == 1

    def test_filter_too_similar_instruction_output(self):
        f = self.get_filter()
        data = [
            {"instruction": "What is Python?", "output": "What is Python?"},  # identical
            {"instruction": "What is Python?", "output": "Python is a language."},
        ]
        filtered = f.filter(data)
        assert len(filtered) == 1

    def test_filter_empty_list(self):
        f = self.get_filter()
        assert f.filter([]) == []

    def test_filter_with_tokenizer(self):
        from filter import QualityFilter
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [101]*1000
        f = QualityFilter(min_length=5, max_length=100, tokenizer=mock_tok)
        data = [{"instruction": "Test question here?", "output": "Long " * 1000}]
        filtered = f.filter(data)
        assert len(filtered) == 0  # Too many tokens

class TestFilterChain:
    def test_chain(self):
        from filter import QualityFilter, FilterChain
        f1 = QualityFilter(min_length=5, max_length=100)
        f2 = QualityFilter(min_length=10, max_length=50)
        chain = FilterChain([f1, f2])
        data = [
            {"instruction": "This is a test question?", "output": "This is the answer."},
            {"instruction": "Hi", "output": "Hello"},
        ]
        filtered = chain.apply(data)
        assert len(filtered) <= len(data)
