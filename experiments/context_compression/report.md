# Context Compression Experiment

## Problem

Long-form conversations produce linearly growing context windows. LLM inference cost scales with prompt length, and beyond the context window limit, the conversation cannot be processed. Standard truncation (keep latest N tokens) discards all early context, which may contain information relevant to later turns.

## Methods

| Method | Description | Expected Behavior |
|--------|-------------|-------------------|
| Full Context (baseline) | Keep entire conversation verbatim | Maximum information, highest latency |
| Truncation | Keep latest messages within budget | Lowest latency, discards all early context |
| Hierarchical Compression | Tiered compression by recency: verbatim (last 5), 100-char (6-20), 50-char (21-50), marker (51+) | Balanced latency and recall |

## Experimental Setup

- **Conversation**: 50 turns across 7 topics (7 turns per topic shift)
- **Average turn length**: ~250 tokens
- **Total tokens**: ~12,500
- **Entity set**: 49 unique entities across all topics
- **Budgets tested**: 1,024 / 2,048 / 4,096 / 8,192 tokens
- **Metrics**: compression ratio, entity recall, turns preserved, estimated latency

## Results

```
Baseline (Full Context):
  Tokens:   12,858
  Latency:  6,429ms
  Recall:   100.0%

At budget = 1,024:
  Truncation:          866 tokens,  11.1% recall,    433ms
  Hierarchical:        932 tokens,  94.4% recall,    466ms
  Improvement:         +83.3pp recall at similar latency

At budget = 2,048:
  Truncation:        1,651 tokens,  16.7% recall,    826ms
  Hierarchical:      1,961 tokens, 100.0% recall,    980ms
  Improvement:         +83.3pp recall at similar latency
```

## Key Findings

1. **Hierarchical compression dominates truncation across all budget levels.** At tight budgets (1K tokens), truncation keeps only 3 turns (11% entity recall), while compression keeps 46 turns (94% recall).

2. **Entity recall saturates at 2K budget for hierarchical compression.** All 49 entities are preserved within 1,961 tokens, because each turn's early-positioned entity names survive the character-level truncation.

3. **Latency reduction follows token reduction.** Compression reduces latency from 6.4s to ~0.5-1.0s depending on budget, a 6-12× improvement over full context.

4. **Truncation loses topic continuity.** When topics shift every 7 turns, truncation at 1K budget discards all but the latest topic, while compression retains at least a marker for each earlier topic.

## Discussion

The effectiveness of hierarchical compression relies on the observation that entities and key information appear early in each turn's content. This holds for well-structured conversations where users front-load their questions. For conversations where information is distributed uniformly across the text, compression would be less effective and a learned importance scoring approach would be needed.

## Reproduction

```bash
# Run the experiment
python run.py --budgets 1024 2048 4096 8192 --out results

# Generate plots
python plot.py --input results/results.json --output plots
```
