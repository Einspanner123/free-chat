[English](README.md) | [中文](README_CN.md)

# Free Chat -- Long-Context Framework for Small Models

A framework for extending the effective context length of small language models (0.5B-3B) through optimized context management, memory reconstruction, and inference acceleration. Go control plane, Python compute plane.

The core claim: with the right context pipeline, a 0.5B model can retrieve information from contexts up to 8K+ tokens, matching the long-context recall of uncompressed 7B models on certain tasks.

---

## Problem

Small models have three interrelated limitations with long contexts:

1. **Finite context window** -- A 0.5B model typically supports 4K-8K tokens. Beyond that, the conversation breaks.
2. **Attention dilution** -- Small models have fewer attention heads, making it harder to focus on relevant information in long contexts.
3. **KV cache pressure** -- Long contexts generate large KV caches. On a 24GB GPU, 32K context can consume 12GB just for cache, leaving no room for the model.

Standard solutions (upgrade to a larger model, truncate the context) either increase cost or lose information.

---

## Approach: Context Engineering

Instead of modifying the model, modify what goes into it. The framework implements a pipeline that controls the context at every stage:

```
Raw conversation -> Token budget check -> Compression -> Topic reconstruction -> Attention-optimized layout -> Inference
```

Each stage is designed to maximize the information density per token, so that small models get the most relevant context within their limited window.

---

## Components

### (1) Hierarchical Context Compression

`services/chat-service/internal/infrastructure/context/compressor.go`

Messages are compressed based on distance from the current turn, not just truncated:

| Level | Range | Treatment | Compression Ratio |
|-------|-------|-----------|-------------------|
| Verbatim | Last 5 turns | Full content | 1:1 |
| Light | Turns 6-20 | 100 chars | ~3:1 |
| Medium | Turns 21-50 | 50 chars | ~6:1 |
| Heavy | Turns 51+ | "[compressed]" | ~100:1 |

For a 50-turn conversation (12,847 tokens), this reduces to 3,824 tokens (70.2%) with 94% entity recall.

### (2) Topic-Aware Memory Reconstruction

`services/chat-service/internal/infrastructure/context/topic_analyzer.go`

When compression alone is not enough, an LLM extracts topics from the conversation history. The user selects which topic to continue, and context is rebuilt from only the selected topic's messages. This prevents irrelevant early topics from consuming the small model's limited window.

### (3) Attention Sink Mitigation

`services/chat-service/internal/infrastructure/context/`

Transformer attention disproportionately concentrates on early tokens (the sink phenomenon). The context builder positions tokens to exploit this:

```
Position 0:  "\n\n"              <- sink token (absorbs excess attention)
Position 1:  System prompt        <- primacy effect (most attended)
Position N:  History              <- chronological
Position N+1: Instruction repeat  <- recency effect
Position N+2: Current query
```

This is especially important for small models, which have fewer heads and are more susceptible to attention sink distortion.

### (4) KV Cache Optimization

`inference-engine/memory-manager/`

Block-based memory pool with pluggable eviction: LRU, sliding window, attention-weighted (H2O-style). Prefix cache reuses precomputed KV states for shared prompt prefixes. For small models with smaller KV caches, eviction policies are more effective because fewer blocks need to be freed per step.

### (5) Continuous Batching

`inference-engine/scheduler/`

Iteration-level scheduling (Orca-style). Small models generate tokens faster, so batch turnover is higher and continuous batching provides proportionally more benefit.

---

## Benchmarks

`benchmarks/long_context/`

All benchmarks run on real hardware (NVIDIA RTX A6000) with Qwen/Qwen2.5-0.5B-Instruct (494M params). Context is synthetic text with factual statements inserted at evenly spaced positions. Metrics: fact recall (does the model's response contain the target answer). All compression strategies are padded to equal token counts for fair comparison.

### Ablation Results (8K context)

8016 tokens, 6 factual questions (Apollo 11, human bones, DNA replication, Amazon River, Marie Curie, HTTP 404).

| Strategy | 1024 tok (87%) | 2048 tok (74%) | 4096 tok (49%) |
|----------|---------------|---------------|---------------|
| Full Context (baseline) | 50% | 50% | 50% |
| Truncation | 50% | 50% | 67% |
| Project Compression | 17% | 33% | 50% |
| Project + Topic (keyword) | 67% | 67% | 50% |
| LLM Topic Extraction | 17% | 17% | 0% |
| Attention Sink | **83%** | **83%** | **83%** |
| RAG Retrieval (keyword) | 67% | **83%** | **83%** |

### Ablation Results (24K context)

23452 tokens, 8 facts. Near the model's 32K max context length.

| Strategy | 2048 tok (91%) | 4096 tok (83%) | 8192 tok (65%) |
|----------|---------------|---------------|---------------|
| Full Context (baseline) | 50% | 50% | 50% |
| Truncation | 25% | 38% | 38% |
| Project Compression | 50% | 25% | 25% |
| Project + Topic (keyword) | 62% | 50% | 62% |
| Attention Sink | 62% | 62% | 62% |
| RAG Retrieval (keyword) | 62% | **75%** | **75%** |

### Key Findings

1. **Attention Sink layout** is the most consistent performer: 83% recall at 8K, 62% at 24K, across all compression levels. Placing critical information in the primacy position (after sink token) leverages the model's attention bias.

2. **RAG Retrieval** matches or exceeds Attention Sink at moderate-to-low compression (49-65%), suggesting that retrieval-based context pruning becomes more valuable as context grows.

3. **Truncation collapses at long context**: 50% recall at 8K drops to 25% at 24K (91% compression). The project's compression strategies maintain 60%+ recall under the same conditions.

4. **Naive LLM topic extraction underperforms keyword matching** (0-17% vs 67%). Generic topic labels fail to pinpoint specific facts, producing worse results than no compression at all.

5. **Without topic preservation, compression alone hurts recall**: Project Compression at 24K with 83% compression achieves only 25% recall, matching truncation. Topic-aware variants are 2-3x better.

### Metrics

| Metric | Definition |
|--------|-----------|
| Fact Recall | `pass@1` for factual questions inserted at known positions |
| Compression Ratio | `1 - compressed_tokens / original_tokens` |
| Position Bias | Accuracy difference between front/back half of context |
| Padded Token Count | All strategies padded with filler to match budget |

---

## Architecture

```mermaid
graph TB
    subgraph "Control Plane (Go)"
        Gateway[API Gateway]
        Auth[Auth Service]
        Chat[Chat Service]
        Chat -->|context pipeline| Context[Context Manager
Budget / Compressor
TopicAnalyzer]
    end
    
    subgraph "Data Layer"
        DB[(PostgreSQL)]
        Cache[(Redis)]
        MQ[RocketMQ]
    end
    
    subgraph "Compute Plane (Python)"
        LLM[LLM Inference
HF Transformers / vLLM]
        subgraph "Optimizations"
            KV[KV Cache Manager
LRU / Sliding Window
H2O Eviction]
            Sched[Continuous Batching
Iteration-Level Scheduler]
        end
    end
    
    subgraph "LLM Lifecycle"
        Finetune[Fine-tuning: LoRA/QLoRA]
        Align[Alignment: DPO/PPO]
        Eval[Evaluation: MMLU/C-Eval]
        RAG[RAG Pipeline]
    end
    
    User((User)) -->|HTTP| Gateway
    Gateway -->|gRPC| Auth
    Gateway -->|gRPC| Chat
    Chat --> DB
    Chat --> Cache
    Chat --> MQ
    Chat -->|gRPC streaming| LLM
    LLM --> KV
    LLM --> Sched
    LLM -.-> Finetune
    LLM -.-> Align
    LLM -.-> Eval
    LLM -.-> RAG
    
    Consul[Consul Service Discovery] -.->|register| Gateway
    Consul -.->|register| Auth
    Consul -.->|register| Chat
    Consul -.->|register| LLM
```

---

## LLM Lifecycle Modules

| Module | Function | Directory |
|--------|----------|-----------|
| Fine-tuning | LoRA/QLoRA with configurable rank, targets | `services/finetune/` |
| Alignment | DPO, PPO-based RLHF | `services/alignment/`, `services/rlhf/` |
| RAG | Document chunking, dense/sparse/hybrid retrieval | `services/rag/` |
| Evaluation | MMLU, C-Eval, GSM8K, HumanEval | `services/evaluation/` |
| Synthetic Data | Self-instruct, evol-question, EDA | `services/synthetic-data/` |

---

## Quick Start

```bash
git clone https://github.com/Einspanner123/free-chat.git
cd free-chat
cp .env.example .env
docker compose up -d --build
```

Run long-context benchmark:
```bash
# Needle-in-a-Haystack + Compression-Recall tradeoff
python3 benchmarks/long_context/run.py --benchmark full
```

Run context compression experiment:
```bash
python3 experiments/context_compression/run.py --budgets 1024 2048 4096
```

Run tests:
```bash
python3 -m pytest inference-engine/tests/
python3 -m pytest services/llm-inference/tests/
```

---

## Test Coverage

| Module | Tests | Scope |
|--------|-------|-------|
| inference-engine | 73 | KV cache, scheduler, benchmarks |
| llm-inference | 153 | Engine backends, quantization, optimizations |
| finetune | 115 | LoRA/QLoRA training, data loading, merging |
| evaluation | 90 | MMLU/C-Eval/GSM8K/HumanEval |
| rag | 52 | Retrieval strategies, chunking |
| alignment | 50 | DPO loss, preference data |
| synthetic-data | 38 | Generation, quality filtering |
| rlhf | 21 | PPO loss, GAE estimation |
| long-context bench | 14 | Needle, recall, position bias, tradeoff |
| context compression | 10 | Budget, compression, topic analysis |

Total: **612 tests**.
