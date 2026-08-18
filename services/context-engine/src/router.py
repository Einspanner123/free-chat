"""
Intent routing: decide *how* to optimize a context.

The project's research shows different tasks need different strategies:
locate → BM25 retrieval, narrative → no compression, QA → topic selection.
This layer classifies the current request and routes it to the strategy that
the benchmark says works for that task.

Rule-based v1. A pluggable IntentClassifier protocol allows an LLM-based
classifier (v2) to replace RuleClassifier without touching the pipeline.
"""

import re
from dataclasses import dataclass
from typing import Protocol

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class Features:
    has_doc_markers: bool  # text contains "Paragraph N:" / numbered sections
    is_long_doc: bool      # text exceeds the long-document threshold
    has_question: bool     # query is a question (EN / ZH)
    descriptive: bool      # query is a substantial statement (locate-by-description)
    multi_turn: bool       # text contains conversation role markers
    narrative_intent: bool  # query asks to generate / write / tell
    synthesis_intent: bool  # query asks to summarize / extract main points
    recency: bool          # query asks about current / recent info


@dataclass
class Intent:
    task: str        # locate | qa | conversation | synthesis | narrative | chat | search
    confidence: float


class IntentClassifier(Protocol):
    """Classify text + query into an intent. Rule-based or LLM-based."""

    def classify(self, text: str, query: str) -> Intent:
        ...


# ---------------------------------------------------------------------------
# Strategy mapping
# ---------------------------------------------------------------------------

# Task -> strategy, derived from benchmark findings:
#   locate       → BM25 top-1 retrieval (98% paragraph localization)
#   qa/synthesis → keyword selection + tiered compression (74%→95%)
#   conversation → attention-sink layout (protects recent turns)
#   narrative    → no compression (research: synthesis suffers under truncation)
#   search       → live web search (recency / knowledge-cutoff questions)
STRATEGY_BY_TASK = {
    "locate": "bm25_top1",
    "qa": "project_topic",
    "conversation": "sink_topic",
    "synthesis": "project_topic",
    "narrative": "full",
    "search": "web_search",
    "chat": "sink_topic",
}

# Below this confidence we do not trust the task label; fall back to the
# safest default strategy instead of risk a wrong one.
_MIN_CONFIDENCE = 0.6


def route(intent: Intent, over_budget: bool) -> str:
    """Map an intent + budget status to a concrete pipeline strategy.

    Guards:
      - search intent  → "web_search" regardless of budget (recency questions
        have no local doc, so the text is short / under budget by definition)
      - not over budget  → "full" (nothing to optimize)
      - low confidence   → "sink_topic" (worst case = safe default)
      - otherwise        → STRATEGY_BY_TASK[intent.task]
    """
    if intent.task == "search" and intent.confidence >= _MIN_CONFIDENCE:
        return "web_search"
    if not over_budget:
        return "full"
    if intent.confidence < _MIN_CONFIDENCE:
        return "sink_topic"
    return STRATEGY_BY_TASK.get(intent.task, "sink_topic")


# ---------------------------------------------------------------------------
# Rule classifier
# ---------------------------------------------------------------------------

_LONG_DOC_CHARS = 800

_WH_QUESTION_RE = re.compile(r"\b(what|which|who|when|where|why|how)\b", re.I)
_CN_QUESTION_RE = re.compile(r"[什么如何哪些谁哪为什么吗是不是多少]")

_PARAGRAPH_RE = re.compile(
    r"(?:paragraph\s*\d+\s*:)|(?:^\s*\d+[\.、)])|(?:^\s*[一二三四五六七八九十]+[、.])",
    re.M | re.I,
)

# Conversation role markers signal a multi-turn history, not a single doc.
_ROLE_RE = re.compile(r"\b(?:user|assistant|human|system|ai):", re.I)

_NARRATIVE_RE = re.compile(
    r"\b(write|tell|generate|create|compose|imagine|story|draft)\b", re.I
)
_SYNTHESIS_RE = re.compile(
    r"\b(summariz|summary|overall|main points?|main idea|key points?|tldr|conclu)\b",
    re.I,
)

# Knowledge-cutoff signal: the answer may postdate the model's training data.
_RECENCY_RE = re.compile(
    r"\b(latest|recent|current|today|now|news|breaking|this (?:week|month|year))\b|"
    r"(最新|现在|最近|今天|今年|当前|本周|本月|昨日|上周|上月|新闻|即将|刚刚)",
    re.I,
)

# 4-digit years from the training-data boundary onward are treated as
# recency signals (e.g. "Who won the 2025 World Cup?").
_RECENT_YEAR_FROM = 2024


def _has_recent_year(query: str) -> bool:
    for m in re.finditer(r"\b(20\d{2})\b", query):
        if int(m.group(1)) >= _RECENT_YEAR_FROM:
            return True
    return False


class RuleClassifier:
    """Deterministic classifier: extract features, then apply the decision chain.

    Decision chain (from the project's benchmark boundary findings):
      1. locate + long doc        → locate / 0.9 (question) or 0.8 (description)
      2. synthesis                → synthesis / 0.7
      3. narrative                → narrative / 0.85
      4. multi-turn               → conversation / 0.7 (doc) or 0.8 (no doc)
      5. recency question, no doc → search / 0.8 (web search, not local QA)
      6. question, long doc       → qa / 0.55 (locate degraded: no doc markers)
      7. question                 → qa / 0.65
      8. fallback                 → chat / 0.3
    """

    def classify(self, text: str, query: str) -> Intent:
        f = self.extract_features(text, query)
        return self._decide(f)

    def extract_features(self, text: str, query: str) -> Features:
        return Features(
            has_doc_markers=bool(_PARAGRAPH_RE.search(text)),
            is_long_doc=len(text) > _LONG_DOC_CHARS,
            has_question=bool(_WH_QUESTION_RE.search(query) or _CN_QUESTION_RE.search(query)),
            # LongBench-style locate queries are statements, not questions.
            descriptive=len(query.strip()) >= 40,
            multi_turn=bool(_ROLE_RE.search(text)),
            narrative_intent=bool(_NARRATIVE_RE.search(query)),
            synthesis_intent=bool(_SYNTHESIS_RE.search(query)),
            recency=bool(_RECENCY_RE.search(query)) or _has_recent_year(query),
        )

    def _decide(self, f: Features) -> Intent:
        # locate-by-description (passage_retrieval) may be phrased as a
        # question OR as a substantial statement describing the answer.
        if f.has_doc_markers and f.is_long_doc and (f.has_question or f.descriptive):
            return Intent("locate", 0.9 if f.has_question else 0.8)
        if f.synthesis_intent:
            return Intent("synthesis", 0.7)
        if f.narrative_intent:
            return Intent("narrative", 0.85)
        if f.multi_turn:
            return Intent("conversation", 0.7 if f.has_doc_markers else 0.8)
        if f.recency and f.has_question and not f.is_long_doc and not f.has_doc_markers:
            # Knowledge-cutoff question with nothing local to answer from →
            # live web search. A provided long doc wins (locate/qa above).
            return Intent("search", 0.8)
        if f.has_question:
            # A question over a markerless long doc looks like locate but there
            # is nothing to retrieve by paragraph → degrade to topic QA.
            return Intent("qa", 0.55 if f.is_long_doc else 0.65)
        return Intent("chat", 0.3)
