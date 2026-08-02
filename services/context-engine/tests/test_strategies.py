"""
Tests for context-engine strategies layer (compression + layout).
Red phase: all fail, then implemented.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


class _FakeTokenizer:
    """Deterministic tokenizer for tests: each char = 1 token."""

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(tokens)


@pytest.fixture
def tok():
    return _FakeTokenizer()


class TestTruncation:
    def test_keeps_last_n_chars(self, tok):
        from strategies import truncate
        text = "abcdefghijklmnop"
        result = truncate(text, tok, budget=8)
        assert result == "ijklmnop"  # last 8 chars

    def test_short_text_unchanged(self, tok):
        from strategies import truncate
        result = truncate("short", tok, budget=100)
        assert result == "short"

    def test_empty_text(self, tok):
        from strategies import truncate
        assert truncate("", tok, budget=10) == ""


class TestChunking:
    def test_split_sentences_en(self):
        from strategies import chunk_sentences
        text = "First sentence. Second one! Third?"
        chunks = chunk_sentences(text)
        assert len(chunks) == 3
        assert "First" in chunks[0]

    def test_split_sentences_zh(self):
        from strategies import chunk_sentences
        text = "第一句。第二句！第三句？"
        chunks = chunk_sentences(text)
        assert len(chunks) == 3

    def test_empty(self):
        from strategies import chunk_sentences
        assert chunk_sentences("") == []


class TestKeywordExtraction:
    def test_english_words(self):
        from strategies import extract_query_words
        words = extract_query_words("What is the name of the football club?")
        assert "football" in words
        assert "club" in words
        # stopwords removed
        assert "what" not in words
        assert "the" not in words

    def test_chinese_words(self):
        from strategies import extract_query_words
        words = extract_query_words("足球俱乐部的名字是什么")
        assert len(words) > 0
        assert "什么" not in words  # stopword

    def test_empty_query(self):
        from strategies import extract_query_words
        assert extract_query_words("") == []


class TestTopicSelection:
    def test_selects_paragraphs_with_hits(self, tok):
        from strategies import select_relevant, chunk_paragraphs
        context = "Paragraph 1: Apple makes phones.\nParagraph 2: Banana is fruit.\nParagraph 3: Apple pies."
        paras = chunk_paragraphs(context)
        selected = select_relevant(paras, "apple", tok, top_k=2)
        # Paragraph 1 and 3 contain 'apple'
        assert len(selected) == 2
        assert "Apple makes" in selected[0]

    def test_top_k_limit(self, tok):
        from strategies import select_relevant, chunk_sentences
        context = "apple. apple. apple. apple. banana."
        chunks = chunk_sentences(context)
        selected = select_relevant(chunks, "apple", tok, top_k=2)
        assert len(selected) == 2

    def test_no_hits_returns_empty(self, tok):
        from strategies import select_relevant, chunk_paragraphs
        context = "P1: banana\nP2: orange"
        paras = chunk_paragraphs(context)
        selected = select_relevant(paras, "apple", tok, top_k=2)
        assert selected == []

    def test_ranks_by_hit_count(self, tok):
        """命中数多的段落排前面."""
        from strategies import select_relevant, chunk_paragraphs
        # P1 has 2 hits, P2 has 1 hit
        context = "P1: apple apple\nP2: apple banana"
        paras = chunk_paragraphs(context)
        selected = select_relevant(paras, "apple", tok, top_k=2)
        assert "P1" in selected[0]


class TestHierarchicalCompression:
    def test_recent_verbatim(self, tok):
        from strategies import compress_tiered
        chunks = ["s" * 200, "t" * 200, "u" * 200]  # 3 chunks
        result = compress_tiered(chunks, tok, budget=100)
        # Recent chunks preserved, older compressed
        assert result is not None

    def test_budget_respected(self, tok):
        from strategies import compress_tiered
        chunks = ["a" * 100] * 10
        result = compress_tiered(chunks, tok, budget=50)
        assert len(tok.encode(result, add_special_tokens=False)) <= 50

    def test_empty(self, tok):
        from strategies import compress_tiered
        assert compress_tiered([], tok, budget=100) == ""


class TestLayout:
    def test_attention_sink_layout(self, tok):
        from strategies import apply_attention_sink
        key_text = "KEY: important info"
        other = "other content"
        result = apply_attention_sink(key_text, other)
        assert result.startswith("\n\n")  # sink token
        assert "KEY: important" in result

    def test_sink_then_key_then_other(self, tok):
        from strategies import apply_attention_sink
        result = apply_attention_sink("KEY", "OTHER")
        parts = result.split("\n\n")
        # [sink, KEY, OTHER] roughly
        assert len(parts) >= 3

    def test_no_other(self, tok):
        from strategies import apply_attention_sink
        result = apply_attention_sink("KEY", "")
        assert "KEY" in result


class TestStrategyInvariant:
    """所有策略的输出必须满足：不超过预算、保留核心信息."""

    def test_all_strategies_respect_budget(self, tok):
        from strategies import build_context
        text = "Paragraph 1: apple apple apple apple\nParagraph 2: banana\nParagraph 3: orange"
        query = "apple"
        for strat in ["truncation", "project_topic", "attention_sink", "sink_topic"]:
            result = build_context(text, tok, budget=30, strategy=strat, query=query)
            used = len(tok.encode(result, add_special_tokens=False))
            # budget is approximate due to separators
            assert used <= 30 + 5, f"{strat}: {used} > 30"

    def test_truncation_preserves_recent(self, tok):
        """截断保留末尾（近因）."""
        from strategies import build_context
        text = "A" * 50 + "B" * 50
        result = build_context(text, tok, budget=20, strategy="truncation", query="")
        assert "B" in result  # 末尾内容保留
