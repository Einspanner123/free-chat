"""
Context strategies: compression and layout primitives.

Pure functions with no state. Each strategy transforms text under a
token budget. High cohesion: every function does one thing.
"""

import re
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_sentences(text: str) -> List[str]:
    """Split text into sentences (supports EN + ZH punctuation)."""
    if not text:
        return []
    return [s for s in re.split(r'(?<=[.!?。！？])\s*', text) if s.strip()]


def chunk_paragraphs(text: str, pattern: str = r'(?=Paragraph \d+:)') -> List[str]:
    """Split text into paragraphs by a boundary pattern."""
    if not text:
        return []
    return [p for p in re.split(pattern, text) if p.strip()]


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_EN_STOPWORDS = {
    'the', 'that', 'this', 'these', 'those', 'with', 'from', 'were', 'have',
    'been', 'their', 'they', 'there', 'about', 'text', 'according', 'article',
    'question', 'answer', 'based', 'following', 'passage', 'main', 'character',
    'summarizes', 'discusses', 'whose', 'name', 'named', 'what', 'which',
    'where', 'when', 'how', 'why', 'who', 'does', 'do', 'did', 'are', 'is',
}

_CN_STOPWORDS = {
    '一个', '什么', '如何', '关于', '根据', '描述', '下列', '其中', '哪些',
    '为什么', '上面', '以下', '文本', '请', '回答', '找出', '匹配', '请根据',
}


def extract_query_words(query: str) -> List[str]:
    """Extract key entities/words from a query (EN + ZH)."""
    if not query:
        return []
    en_words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', query)
                if w.lower() not in _EN_STOPWORDS]
    cn_words = [w for w in re.findall(r'[\u4e00-\u9fa5]{2,6}', query)
                if w not in _CN_STOPWORDS][:5]
    return en_words + cn_words


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def truncate(text: str, tokenizer, budget: int) -> str:
    """Keep the last budget tokens (recency)."""
    if not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= budget:
        return text
    return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Topic selection (score paragraphs by query-word hits)
# ---------------------------------------------------------------------------


def select_relevant(chunks: List[str], query: str, tokenizer,
                    top_k: int = 3) -> List[str]:
    """Score chunks by query-word hits, return top-k most relevant.

    Args:
        chunks: List of text chunks (paragraphs/sentences).
        query: Query string.
        tokenizer: Tokenizer for budget accounting (unused in selection).
        top_k: Number of top chunks to keep.

    Returns:
        Top-k chunks sorted by relevance (descending hits).
    """
    if not chunks or not query:
        return []
    query_words = extract_query_words(query)
    if not query_words:
        return []
    scored = []
    for chunk in chunks:
        hits = sum(1 for w in query_words if w.lower() in chunk.lower())
        if hits > 0:
            scored.append((hits, chunk))
    scored.sort(key=lambda x: -x[0])
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# Hierarchical compression
# ---------------------------------------------------------------------------


def compress_tiered(chunks: List[str], tokenizer, budget: int,
                    levels: Tuple[int, int, int] = (5, 20, 50),
                    max_chars: Tuple[int, int] = (100, 50)) -> str:
    """Compress chunks by recency: recent verbatim, older truncated.

    Args:
        chunks: Chunks ordered oldest -> newest.
        tokenizer: Tokenizer for budget accounting.
        budget: Token budget.
        levels: (verbatim_count, light_count, medium_count).
        max_chars: (light_max_chars, medium_max_chars).

    Returns:
        Compressed text fitting approximately within budget.
    """
    if not chunks:
        return ""
    verbatim_n, light_n, medium_n = levels
    light_chars, medium_chars = max_chars

    result = []
    total = 0
    for i, chunk in enumerate(reversed(chunks)):
        turn = i + 1  # 1 = newest
        if turn <= verbatim_n:
            ct = chunk
        elif turn <= light_n:
            ct = chunk[:light_chars]
        elif turn <= medium_n:
            ct = chunk[:medium_chars]
        else:
            ct = ""
        if not ct:
            continue
        nt = len(tokenizer.encode(ct, add_special_tokens=False))
        if total + nt <= budget:
            result.insert(0, ct)
            total += nt
        else:
            break
    return " ".join(result)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def apply_attention_sink(key_text: str, other_text: str) -> str:
    """Layout: sink token → key info → other content.

    Position 0: "\\n\\n" (sink, absorbs excess attention)
    Position 1: key_text (primacy effect)
    Position N: other_text

    Args:
        key_text: Critical information to place at position 1.
        other_text: Secondary content.

    Returns:
        Laid-out context string.
    """
    if other_text:
        return "\n\n" + key_text + "\n\n" + other_text
    return "\n\n" + key_text


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


def build_context(text: str, tokenizer, budget: int, strategy: str,
                  query: str = "") -> str:
    """Build a context string under a strategy.

    Args:
        text: Source text.
        tokenizer: Tokenizer for budget accounting.
        budget: Token budget.
        strategy: One of truncation / project_topic / attention_sink /
                  sink_topic / bm25_top1.
        query: Query for relevance-based strategies.

    Returns:
        Context string within budget.

    Raises:
        ValueError: Unknown strategy.
    """
    if strategy == "truncation":
        return truncate(text, tokenizer, budget)

    if strategy in ("project_topic", "attention_sink", "sink_topic"):
        chunks = chunk_sentences(text)
        if not chunks:
            return text[:budget] if len(text) > budget else text
        key = select_relevant(chunks, query, tokenizer)
        other = [c for c in chunks if c not in key]

        # Key chunks must also respect budget (avoid over-budget bug)
        key_text = " ".join(key)
        key_tokens = len(tokenizer.encode(key_text, add_special_tokens=False))
        if key_tokens > budget:
            key_text = truncate(key_text, tokenizer, budget)
            key_tokens = budget
        remaining = budget - key_tokens
        compressed_other = compress_tiered(other, tokenizer, remaining, levels=(5, 20, 50))

        if strategy == "project_topic":
            # key + compressed other, key first
            return (key_text + " " + compressed_other).strip()

        # attention_sink / sink_topic: key at position 1 after sink
        return apply_attention_sink(key_text, compressed_other)

    if strategy == "bm25_top1":
        # Requires a retriever; delegate to pipeline layer.
        raise NotImplementedError(
            "bm25_top1 strategy requires pipeline-level retrieval. "
            "Use ContextPipeline instead."
        )

    raise ValueError(f"Unknown strategy: {strategy}")
