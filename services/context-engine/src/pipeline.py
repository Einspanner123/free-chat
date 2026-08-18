"""
Pipeline orchestration: retrieval → compression → layout → assembly.

Composes the retriever layer and strategies layer into a single
context-building pipeline. High cohesion: pipeline only orchestrates.
"""

from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional

from strategies import (
    build_context, truncate, chunk_paragraphs, select_relevant,
    compress_tiered, apply_attention_sink,
)
from retriever import RetrieverFactory, BaseContextRetriever
from router import IntentClassifier, RuleClassifier, route
from search.client import WebSearchClient


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    strategy: str = "auto"
    budget: int = 1024
    retriever: str = "bm25"
    top_k: int = 1
    chunk_pattern: str = r'(?=Paragraph \d+:)'
    classifier: Optional[IntentClassifier] = None
    # Web search knobs (strategy="web_search" / auto-routed search intent).
    search_provider: Optional[str] = None     # provider name, or None = first available
    web_top_k: int = 3                        # results to assemble into context
    search_client: Optional[WebSearchClient] = None  # injectable for tests


def _format_web_results(results: List[Dict]) -> str:
    """Assemble search hits into a "Sources:" context block."""
    parts = ["Sources:"]
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        desc = (r.get("description") or "").strip()
        body = f"[{i}] {title}\n{url}" if title else f"[{i}] {url}"
        if desc:
            body += f"\n{desc}"
        parts.append(body)
    return "\n\n".join(parts)


class ContextPipeline:
    """Build an optimized context under a token budget.

    Strategy behaviors:
      - truncation:      keep last budget tokens
      - project_topic:   keyword-select relevant chunks + tiered compression
      - attention_sink:  key chunks at position 1 (sink) + compression
      - sink_topic:      same as attention_sink (combined)
      - bm25_top1:       BM25 retrieve top-1 paragraph (RAG)
      - keyword_top1:    keyword retrieve top-1 paragraph
      - full:            identity, no compression (narrative / under budget)
      - auto:            classify intent → route to the best strategy
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._classifier: IntentClassifier = config.classifier or RuleClassifier()
        self._retriever: Optional[BaseContextRetriever] = None
        if config.strategy in ("bm25_top1", "keyword_top1"):
            rname = "bm25" if config.strategy == "bm25_top1" else "keyword"
            self._retriever = RetrieverFactory.create(rname)

    def build(self, text: str, tokenizer, query: str = "") -> str:
        """Build a context string under the configured strategy.

        Args:
            text: Source text.
            tokenizer: Tokenizer for budget accounting.
            query: Query for relevance-based strategies.

        Returns:
            Context string within budget.

        Raises:
            ValueError: Unknown strategy.
        """
        result = self.build_with_metadata(text, tokenizer, query)
        return result["context"]

    def build_with_metadata(self, text: str, tokenizer, query: str = "") -> Dict:
        """Build context and return metadata (strategy, tokens, ratio)."""
        strat = self.config.strategy
        budget = self.config.budget
        full_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        # Identity: keep the full text (narrative / under budget)
        if strat == "full":
            return {
                "context": text, "strategy": "full",
                "tokens": full_tokens, "compression_ratio": 0.0,
            }

        # Auto: classify intent → route to the best strategy for this task
        if strat == "auto":
            intent = self._classifier.classify(text, query)
            resolved = route(intent, over_budget=full_tokens > budget)
            meta = {"routed_from": "auto", "intent": intent.task,
                    "confidence": round(intent.confidence, 4)}
            if resolved == "full":
                return {
                    "context": text, "strategy": "full",
                    "tokens": full_tokens, "compression_ratio": 0.0, **meta,
                }
            sub = ContextPipeline(replace(self.config, strategy=resolved))
            return {**sub.build_with_metadata(text, tokenizer, query), **meta}

        # RAG retrieval strategies
        if strat in ("bm25_top1", "keyword_top1"):
            if self._retriever is None:
                raise ValueError(f"Retriever not initialized for {strat}")
            paras = chunk_paragraphs(text, self.config.chunk_pattern)
            docs = []
            for p in paras:
                m = __import__('re').match(r'(?:Paragraph )?(\d+)?:?', p)
                docs.append({"id": f"chunk_{len(docs)}", "text": p})
            self._retriever.index(docs)
            results = self._retriever.retrieve(query, k=self.config.top_k)
            ctx = self._retriever.format_results(results, docs)
            # 检索空结果 → 回退截断（recency），保证上下文永不为空
            if not ctx:
                ctx = truncate(text, tokenizer, budget)
            # Compress if over budget
            if len(tokenizer.encode(ctx, add_special_tokens=False)) > budget:
                ctx = truncate(ctx, tokenizer, budget)
            used = len(tokenizer.encode(ctx, add_special_tokens=False))
            return {
                "context": ctx, "strategy": strat,
                "tokens": used,
                "compression_ratio": round(1 - used / full_tokens, 4) if full_tokens else 0,
            }

        # Live web search (recency / knowledge-cutoff questions)
        if strat == "web_search":
            client = self.config.search_client or WebSearchClient(provider=self.config.search_provider)
            results = client.search(query, limit=self.config.web_top_k)
            if not results:
                # No provider / no hits → degrade to the safe default so the
                # context is never empty and auto-routing never errors.
                sub = ContextPipeline(replace(self.config, strategy="sink_topic"))
                return sub.build_with_metadata(text, tokenizer, query)
            ctx = _format_web_results(results)
            if len(tokenizer.encode(ctx, add_special_tokens=False)) > budget:
                ctx = truncate(ctx, tokenizer, budget)
            used = len(tokenizer.encode(ctx, add_special_tokens=False))
            return {
                "context": ctx, "strategy": strat, "tokens": used,
                "compression_ratio": round(1 - used / full_tokens, 4) if full_tokens else 0,
                "source": "web",
            }

        # Strategy-layer paths
        if strat in ("truncation", "project_topic", "attention_sink", "sink_topic"):
            ctx = build_context(text, tokenizer, budget, strat, query)
            used = len(tokenizer.encode(ctx, add_special_tokens=False))
            return {
                "context": ctx, "strategy": strat,
                "tokens": used,
                "compression_ratio": round(1 - used / full_tokens, 4) if full_tokens else 0,
            }

        raise ValueError(f"Unknown strategy: {strat}")
