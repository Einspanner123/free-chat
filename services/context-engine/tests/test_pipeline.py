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
