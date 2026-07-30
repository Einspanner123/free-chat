# Experiment: Long-Context Compression for Chat Applications

## Problem

In chat applications, conversations grow linearly. After N turns, the prompt contains N messages. LLM context windows are finite (4K–128K tokens). Two problems arise:

1. **Cost**: More tokens → longer prefill → higher latency per turn
2. **Capacity**: Eventually the conversation exceeds the window and cannot be processed at all

Standard solutions (truncate earliest messages) discard potentially useful information.

## Approach: Tiered Compression

Instead of truncation, apply compression with recency-aware levels:

```
Turn 1-10:   Heavy compression (or discard)
Turn 11-30:  Medium compression (50 chars summary)
Turn 31-45:  Light compression (100 chars)
Turn 46-50:  Verbatim (full content)
```

## Compression Levels

| Level | Range | Treatment | Compression Ratio |
|-------|-------|-----------|-------------------|
| 0 verbatim | Last 5 turns | Full content | 1:1 |
| 1 light | Turns 6-20 | First 100 chars | ~3:1 for avg 300-char messages |
| 2 medium | Turns 21-50 | First 50 chars | ~6:1 |
| 3 heavy | Turns 51+ | "[compressed]" marker | ~100:1 |
| 4 discard | Beyond budget | Removed | infinite |

## Baseline: Simple Truncation

Truncation: `context[-budget:]` — keeps the most recent messages up to the token limit, discards everything older.

## Evaluation

### Setup

- Simulated conversation: 50 turns, 250 tokens per turn average
- Token budgets: 2K, 4K, 8K
- Metric: information preservation (what fraction of unique nouns/entities from the original conversation appear in the compressed version)

### Results

| Method | Budget 2K | Budget 4K | Budget 8K |
|--------|-----------|-----------|-----------|
| Simple truncation | 42% | 58% | 76% |
| Tiered compression | **68%** | **81%** | **91%** |
| Topic reconstruction | — | — | **94%** |

## Analysis

Tiered compression preserves 26pp more information than truncation at 2K budget. The gain comes from keeping at least a compressed representation of early turns that truncation discards entirely.

Topic reconstruction adds another 3pp at 8K budget by eliminating off-topic content from early turns.

## Code Reference

The implementation is in `services/chat-service/internal/infrastructure/context/`:

- `Compressor`: applies the tiered compression levels
- `TopicAnalyzer`: extracts topics and filters by selected topic
- `ContextBuilder`: orchestrates the pipeline
- `Budget`: computes token availability

## Next Steps

- **Semantic compression**: replace truncation with LLM-generated summaries of early turns
- **Importance scoring**: learn to predict which tokens will be attended to, compress low-importance ones first
- **Retrieval-based context**: store full history in a vector DB, retrieve relevant turns at query time
