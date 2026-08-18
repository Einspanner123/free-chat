"""
Tests for the context-engine intent router.

The router decides *how* to optimize a context based on the nature of the
query + source text. It exists because the project's own research shows
different tasks need different strategies (locate → BM25, narrative → no
compression, QA → topic selection).

Red phase: all fail, then implemented.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


class TestRoute:
    """route() maps an Intent + budget status to a concrete strategy."""

    def test_not_over_budget_returns_full(self):
        from router import Intent, route
        assert route(Intent(task="locate", confidence=0.9), over_budget=False) == "full"

    def test_low_confidence_returns_sink_topic(self):
        from router import Intent, route
        assert route(Intent(task="locate", confidence=0.5), over_budget=True) == "sink_topic"

    def test_locate_maps_to_bm25_top1(self):
        from router import Intent, route
        assert route(Intent(task="locate", confidence=0.9), over_budget=True) == "bm25_top1"

    def test_qa_maps_to_project_topic(self):
        from router import Intent, route
        assert route(Intent(task="qa", confidence=0.65), over_budget=True) == "project_topic"

    def test_conversation_maps_to_sink_topic(self):
        from router import Intent, route
        assert route(Intent(task="conversation", confidence=0.8), over_budget=True) == "sink_topic"

    def test_synthesis_maps_to_project_topic(self):
        from router import Intent, route
        assert route(Intent(task="synthesis", confidence=0.7), over_budget=True) == "project_topic"

    def test_narrative_maps_to_full(self):
        from router import Intent, route
        assert route(Intent(task="narrative", confidence=0.85), over_budget=True) == "full"

    def test_unknown_task_defaults_to_sink_topic(self):
        from router import Intent, route
        assert route(Intent(task="unknown", confidence=0.9), over_budget=True) == "sink_topic"

    def test_search_maps_to_web_search_even_under_budget(self):
        # Recency questions have no local doc, so the text is short / not over
        # budget — the usual "not over_budget → full" guard must not apply.
        from router import Intent, route
        assert route(Intent(task="search", confidence=0.8), over_budget=False) == "web_search"

    def test_search_maps_to_web_search_over_budget(self):
        from router import Intent, route
        assert route(Intent(task="search", confidence=0.8), over_budget=True) == "web_search"

    def test_search_low_confidence_falls_back_to_sink_topic(self):
        from router import Intent, route
        assert route(Intent(task="search", confidence=0.5), over_budget=True) == "sink_topic"


class TestRuleClassifier:
    """Deterministic classifier encoding the research-derived decision chain."""

    def test_locate_long_doc(self):
        from router import RuleClassifier
        text = "Paragraph 1: The first section.\nParagraph 2: The second section.\n" * 50
        intent = RuleClassifier().classify(text, "Which paragraph matches this description?")
        assert intent.task == "locate"
        assert intent.confidence >= 0.9

    def test_locate_descriptive_statement(self):
        # LongBench passage_retrieval_en queries are statements, not questions.
        # A long descriptive query (no wh-words) over a marked long doc is locate.
        from router import RuleClassifier
        text = "Paragraph 1: The first section.\nParagraph 2: The second section.\n" * 50
        query = ("These passages describe several notable achievements during the season. "
                 "Michigan reached its best conference record since the late seventies.")
        intent = RuleClassifier().classify(text, query)
        assert intent.task == "locate"
        assert intent.confidence >= 0.8

    def test_locate_without_doc_markers_degrades_to_qa(self):
        from router import RuleClassifier
        text = "Long text without any paragraph markers. " * 200
        intent = RuleClassifier().classify(text, "Which part describes the discovery?")
        assert intent.task == "qa"
        assert intent.confidence < 0.9

    def test_synthesis(self):
        from router import RuleClassifier
        text = "A" * 3000
        intent = RuleClassifier().classify(text, "Summarize the main points of the article.")
        assert intent.task == "synthesis"
        assert intent.confidence >= 0.7

    def test_narrative(self):
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "Write a short story about a dragon.")
        assert intent.task == "narrative"

    def test_multi_turn_conversation(self):
        from router import RuleClassifier
        text = "user: Hi\nassistant: Hello\nuser: What did I ask you before?"
        intent = RuleClassifier().classify(text, "What did I ask you before?")
        assert intent.task == "conversation"
        assert intent.confidence >= 0.7

    def test_single_question_qa(self):
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "What is the capital of France?")
        assert intent.task == "qa"

    def test_chinese_question_qa(self):
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "北京的人口有多少？")
        assert intent.task == "qa"

    def test_fallback_chat_low_confidence(self):
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "hello there")
        assert intent.task == "chat"
        assert intent.confidence < 0.6  # worst case → sink_topic guard

    def test_recency_question_empty_text_routes_to_search(self):
        # "What's the latest..." over an empty local doc is a knowledge-cutoff
        # question — it needs web search, not local QA.
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "What is the latest news about AI?")
        assert intent.task == "search"
        assert intent.confidence >= 0.8

    def test_recency_question_recent_year_routes_to_search(self):
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "Who won the 2025 World Cup?")
        assert intent.task == "search"

    def test_chinese_recency_question_routes_to_search(self):
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "今天上海的天气怎么样？")
        assert intent.task == "search"

    def test_recency_question_with_long_doc_stays_local(self):
        # A provided long doc wins: don't leave it to search the web.
        from router import RuleClassifier
        text = "Paragraph 1: The first section.\nParagraph 2: The second section.\n" * 50
        intent = RuleClassifier().classify(text, "What is the latest change in paragraph 2?")
        assert intent.task == "locate"

    def test_non_recency_question_stays_qa(self):
        from router import RuleClassifier
        intent = RuleClassifier().classify("", "What is the capital of France?")
        assert intent.task == "qa"
