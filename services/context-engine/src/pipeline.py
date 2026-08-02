"""
Pipeline orchestration: retrieval → compression → layout → assembly.

Composes the retriever layer and strategies layer into a single
context-building pipeline. High cohesion: pipeline only orchestrates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from strategies import (
    build_context, truncate, chunk_paragraphs, select_relevant,
    compress_tiered, apply_attention_sink,
)
from retriever import RetrieverFactory, BaseContextRetriever


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    strategy: str = "truncation"
    budget: int = 1024
    retriever: str = "bm25"
    top_k: int = 1
    chunk_pattern: str = r'(?=Paragraph \d+:)'


class ContextPipeline:
    """Build an optimized context under a token budget.

    Strategy behaviors:
      - truncation:      keep last budget tokens
      - project_topic:   keyword-select relevant chunks + tiered compression
      - attention_sink:  key chunks at position 1 (sink) + compression
      - sink_topic:      same as attention_sink (combined)
      - bm25_top1:       BM25 retrieve top-1 paragraph (RAG)
      - keyword_top1:    keyword retrieve top-1 paragraph
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
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
            # Compress if over budget
            if len(tokenizer.encode(ctx, add_special_tokens=False)) > budget:
                ctx = truncate(ctx, tokenizer, budget)
            used = len(tokenizer.encode(ctx, add_special_tokens=False))
            return {
                "context": ctx, "strategy": strat,
                "tokens": used,
                "compression_ratio": round(1 - used / full_tokens, 4) if full_tokens else 0,
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
