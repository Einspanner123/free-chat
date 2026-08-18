"""
Tests for context-engine pipeline layer.
Red phase: all fail, then implemented.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(tokens)


@pytest.fixture
def tok():
    return _FakeTokenizer()


class TestPipeline:
    def test_pipeline_build_context(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        config = PipelineConfig(strategy="truncation", budget=50)
        pipe = ContextPipeline(config)
        ctx = pipe.build("A" * 100, tok, query="")
        assert len(tok.encode(ctx, add_special_tokens=False)) <= 55

    def test_pipeline_with_query_topic(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        config = PipelineConfig(strategy="project_topic", budget=30)
        pipe = ContextPipeline(config)
        text = "Paragraph 1: apple apple apple\nParagraph 2: banana"
        ctx = pipe.build(text, tok, query="apple")
        assert "apple" in ctx  # 相关段落保留

    def test_pipeline_sink(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        config = PipelineConfig(strategy="attention_sink", budget=30)
        pipe = ContextPipeline(config)
        text = "apple content here apple apple"
        ctx = pipe.build(text, tok, query="apple")
        assert ctx.startswith("\n\n")  # sink token

    def test_pipeline_bm25(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        config = PipelineConfig(strategy="bm25_top1", budget=50)
        pipe = ContextPipeline(config)
        text = "Paragraph 1: Apple makes phones.\nParagraph 2: Banana fruit.\nParagraph 3: Apple pies."
        ctx = pipe.build(text, tok, query="Apple makes phones")
        # BM25 检索 top-1 → 应该是 Paragraph 1
        assert "Apple makes" in ctx

    def test_pipeline_invalid_strategy(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        config = PipelineConfig(strategy="invalid", budget=50)
        pipe = ContextPipeline(config)
        with pytest.raises(ValueError):
            pipe.build("text", tok, query="")

    def test_pipeline_returns_metadata(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        config = PipelineConfig(strategy="project_topic", budget=30)
        pipe = ContextPipeline(config)
        result = pipe.build_with_metadata("apple banana", tok, query="apple")
        assert "context" in result
        assert "strategy" in result
        assert "tokens" in result
        assert "compression_ratio" in result


class TestPipelineE2E:
    def test_retrieval_then_compression(self, tok):
        """管道 = 检索 top-1 + 若超预算则压缩."""
        from pipeline import ContextPipeline, PipelineConfig
        config = PipelineConfig(strategy="bm25_top1", budget=100)
        pipe = ContextPipeline(config)
        text = "Paragraph 1: " + "A" * 200 + "\nParagraph 2: banana\nParagraph 3: orange"
        ctx = pipe.build(text, tok, query="banana orange")
        # 检索到 banana/orange 段落，压缩到预算
        assert len(tok.encode(ctx, add_special_tokens=False)) <= 105
        assert "banana" in ctx or "orange" in ctx

    def test_all_strategies_work_with_pipeline(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        text = "Paragraph 1: apple apple\nParagraph 2: banana\nParagraph 3: cherry"
        for strat in ["truncation", "project_topic", "attention_sink", "sink_topic", "bm25_top1"]:
            config = PipelineConfig(strategy=strat, budget=50)
            pipe = ContextPipeline(config)
            ctx = pipe.build(text, tok, query="apple")
            assert ctx is not None
            assert len(tok.encode(ctx, add_special_tokens=False)) <= 55


class TestPipelineAuto:
    def test_auto_not_over_budget_returns_full(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        pipe = ContextPipeline(PipelineConfig(strategy="auto", budget=1000))
        result = pipe.build_with_metadata("short text", tok, query="")
        assert result["strategy"] == "full"
        assert result["context"] == "short text"

    def test_auto_narrative_routes_to_full(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        pipe = ContextPipeline(PipelineConfig(strategy="auto", budget=10))
        result = pipe.build_with_metadata("A" * 100, tok, query="Write a story about X.")
        assert result["strategy"] == "full"
        assert result["context"] == "A" * 100

    def test_auto_locate_routes_to_bm25(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        text = "Paragraph 1: apple apple apple.\n" * 40
        pipe = ContextPipeline(PipelineConfig(strategy="auto", budget=100))
        result = pipe.build_with_metadata(text, tok, query="Which paragraph mentions apple?")
        assert result["strategy"] == "bm25_top1"
        assert result["intent"] == "locate"
        assert result["confidence"] >= 0.9

    def test_auto_locate_descriptive_routes_to_bm25(self, tok):
        # Real LongBench passage_retrieval queries are statements, not "?".
        from pipeline import ContextPipeline, PipelineConfig
        text = "Paragraph 1: apple apple apple.\n" * 40
        pipe = ContextPipeline(PipelineConfig(strategy="auto", budget=100))
        result = pipe.build_with_metadata(
            text, tok,
            query="The text discusses apples in detail. Apples grow on trees and are made into pies.",
        )
        assert result["strategy"] == "bm25_top1"
        assert result["intent"] == "locate"

    def test_auto_low_confidence_falls_back_to_sink(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        pipe = ContextPipeline(PipelineConfig(strategy="auto", budget=10))
        result = pipe.build_with_metadata("A" * 100, tok, query="hello there")
        assert result["strategy"] == "sink_topic"
        assert result["routed_from"] == "auto"

    def test_bm25_empty_retrieval_falls_back(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        # query shares no terms with the doc → BM25 returns nothing.
        # Context must never be empty: fall back to truncation (recency).
        config = PipelineConfig(strategy="bm25_top1", budget=50)
        pipe = ContextPipeline(config)
        text = "Paragraph 1: zzz qqq xxx.\nParagraph 2: yyy wwww."
        ctx = pipe.build(text, tok, query="apple banana cherry")
        assert ctx != ""
        assert len(tok.encode(ctx, add_special_tokens=False)) <= 55

    def test_full_strategy_is_identity(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        pipe = ContextPipeline(PipelineConfig(strategy="full", budget=5))
        result = pipe.build_with_metadata("some long text here", tok, query="")
        assert result["context"] == "some long text here"
        assert result["compression_ratio"] == 0.0

    def test_custom_classifier_injectable(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        from router import Intent

        class FakeClassifier:
            def classify(self, text, query):
                return Intent(task="qa", confidence=0.9)

        pipe = ContextPipeline(PipelineConfig(
            strategy="auto", budget=10, classifier=FakeClassifier(),
        ))
        result = pipe.build_with_metadata("A" * 100, tok, query="anything")
        assert result["strategy"] == "project_topic"


class _FakeSearchClient:
    """Stand-in for WebSearchClient: returns a fixed hit list."""

    def __init__(self, hits=None):
        self._hits = hits if hits is not None else [
            {"title": "AI News", "url": "https://example.com/ai",
             "description": "latest AI breakthroughs", "position": 1},
        ]
        self.last_query = None

    def search(self, query, limit=5):
        self.last_query = query
        return self._hits


class TestPipelineWebSearch:
    def test_web_search_strategy_builds_sources_context(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        pipe = ContextPipeline(PipelineConfig(
            strategy="web_search", budget=500, search_client=_FakeSearchClient(),
        ))
        result = pipe.build_with_metadata("", tok, query="latest AI news")
        assert result["strategy"] == "web_search"
        assert result["source"] == "web"
        assert "Sources:" in result["context"]
        assert "AI News" in result["context"]

    def test_web_search_passes_query_through(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        client = _FakeSearchClient()
        pipe = ContextPipeline(PipelineConfig(strategy="web_search", budget=500, search_client=client))
        pipe.build("", tok, query="today weather shanghai")
        assert client.last_query == "today weather shanghai"

    def test_web_search_truncates_to_budget(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        big = [{"title": "T", "url": "http://x.com", "description": "word " * 200, "position": 1}]
        pipe = ContextPipeline(PipelineConfig(
            strategy="web_search", budget=30, search_client=_FakeSearchClient(hits=big),
        ))
        result = pipe.build_with_metadata("", tok, query="q")
        assert 0 < len(tok.encode(result["context"], add_special_tokens=False)) <= 35

    def test_web_search_empty_results_degrades_to_sink_topic(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        pipe = ContextPipeline(PipelineConfig(
            strategy="web_search", budget=50, search_client=_FakeSearchClient(hits=[]),
        ))
        result = pipe.build_with_metadata("Some local text here.", tok, query="anything")
        assert result["strategy"] == "sink_topic"
        assert result["context"] != ""

    def test_auto_recency_routes_to_web_search(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        pipe = ContextPipeline(PipelineConfig(
            strategy="auto", budget=1000, search_client=_FakeSearchClient(),
        ))
        result = pipe.build_with_metadata("", tok, query="What is the latest news about AI?")
        assert result["intent"] == "search"
        assert result["routed_from"] == "auto"
        assert result["strategy"] == "web_search"
        assert "Sources:" in result["context"]

    def test_auto_recency_without_provider_degrades_gracefully(self, tok):
        from pipeline import ContextPipeline, PipelineConfig
        # No search_client injected and no ddgs installed → no provider →
        # web_search falls back to sink_topic instead of erroring.
        pipe = ContextPipeline(PipelineConfig(strategy="auto", budget=1000))
        result = pipe.build_with_metadata("", tok, query="What is the latest news about AI?")
        assert result["routed_from"] == "auto"
        assert result["strategy"] == "sink_topic"
