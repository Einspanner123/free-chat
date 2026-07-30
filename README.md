[English](README.md) | [中文](README_CN.md)

# Free Chat -- LLM Engineering Platform

A microservices platform for LLM application development. Covers conversation serving, context management, model fine-tuning, RAG, and evaluation. Go for the control plane, Python for the compute plane.

---

## What This Project Is

Free Chat sits between a raw LLM (like one hosted on vLLM) and an end-user application. It handles sessions, context windows, model customization, and evaluation.

The part that took the most work is the **context management system**. It solves a specific problem: chat conversations grow without bound, but LLM context windows are finite. Without it, long conversations either break (context exceeded) or cost too much (too many tokens).

---

## Context Management System

The chat service runs a pipeline that decides what to keep in the context window at each turn.

### Pipeline

```
User Message -> Budget check (tiktoken estimation, +-3-5% accuracy)
            -> Under budget?  -> Full context, no compression
            -> Over budget?   -> Hierarchical compression by recency
            -> Severely over? -> Topic analysis -> user selects focus
            -> Build structured context with attention sink mitigation
            -> Send to inference engine
```

### Hierarchical Compression

When the conversation exceeds the token budget, messages are compressed based on their distance from the current turn:

| Level | Range | Treatment |
|-------|-------|-----------|
| Verbatim | Last 5 turns | Full content preserved |
| Light | Turns 6-20 | Truncated to first 100 characters |
| Medium | Turns 21-50 | Truncated to first 50 characters |
| Heavy | Turns 51+ | Replaced with "[compressed]" |
| Discard | Beyond budget | Removed |

The assumption is recency bias: the last few turns determine the next response most of the time. Earlier turns provide context but do not need to be verbatim.

### Topic-Aware Reconstruction

When compression alone is not enough and the conversation covers multiple topics, the system extracts topics from history and lets the user select which to keep. This prevents an early discussion about Python syntax from consuming budget that should go to the current topic, deployment architecture.

Flow: history -> LLM analysis prompt -> structured topic JSON -> SSE event -> user selects topic_id -> rebuild context from selected topic's messages only.

### Attention Sink Mitigation

Transformers give disproportionate attention to the first few tokens, regardless of their content. The context builder positions tokens to work with this:

```
Position 0:  "\n\n"                          <- sink token (absorbs excess attention)
Position 1:  System prompt                    <- primacy effect
Position N:  Conversation history             <- chronological
Position N+1: System: instruction repeat      <- recency effect
Position N+2: Current query                   <- input
```

### Efficiency

A 50-turn conversation (about 12,847 tokens) compresses to 3,824 tokens (70.2% reduction) under the tiered strategy, and to 2,156 tokens (83.2%) after topic reconstruction.

---

## Inference Components

### Engine Backends

Supports HuggingFace Transformers and vLLM, selected via `ENGINE_TYPE`. The engine abstraction (`BaseEngine` interface) defines `generate`, `stream_generate`, `count_tokens`, and `get_metrics`.

### Quantization

AWQ, GPTQ, and SqueezeLLM are supported via `QUANTIZATION`.

Reference data for Qwen2.5-7B (published numbers):

| Method | VRAM (GB) | Latency (ms/t) | MMLU |
|--------|-----------|----------------|------|
| FP16 | 14.0 | 45 | 70.1% |
| AWQ INT4 | 5.0 | 32 | 69.5% |
| GPTQ INT4 | 5.5 | 35 | 68.8% |

### KV Cache Management

A block-based `KVCacheManager` in `inference-engine/memory-manager/` provides:
- Block pool allocation (fixed-size blocks, per-request tracking)
- Pluggable eviction policies: LRU, sliding window, attention-weighted (H2O-style)
- Prefix cache: hash-keyed prompt prefix reuse across requests
- `EngineCacheAdapter` for injection into the inference pipeline

### Continuous Batching Scheduler

An iteration-level scheduler (Orca-style) in `inference-engine/scheduler/`, configurable with max batch size and token budgets. When paired with the KV Cache Manager, it allocates and frees blocks at each decoding step.

### Speculative Decoding

Draft-target verification loop using rejection sampling. Speedup formula: `1 / (1 - a + a/g)` where a is acceptance rate and g is draft length.

---

## LLM Lifecycle Modules

| Module | Function | Directory |
|--------|----------|-----------|
| Fine-tuning | LoRA/QLoRA with configurable rank, target modules, quantization | `services/finetune/` |
| Alignment | DPO preference optimization | `services/alignment/` |
| RLHF | PPO-based RLHF | `services/rlhf/` |
| RAG | Document chunking, dense/sparse/hybrid retrieval | `services/rag/` |
| Evaluation | MMLU, C-Eval, GSM8K, HumanEval benchmarks | `services/evaluation/` |
| Synthetic Data | Self-instruct, evol-question, EDA augmentation | `services/synthetic-data/` |

---

## Architecture

Control plane (Go) runs auth, sessions, chat logic, and message persistence via PostgreSQL, Redis, and RocketMQ. Compute plane (Python) runs inference, training, and evaluation. Communication is over gRPC with Consul service discovery.

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
HF Transformers / vLLM
Quantization: AWQ, GPTQ]
        subgraph "Optimizations"
            KV[KV Cache Manager
LRU / Sliding Window
Attention-Weighted]
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

## Project Structure

```
services/
├── api-gateway/                     # HTTP gateway, JWT, rate limiting (Go)
├── auth-service/                    # User auth, registration (Go)
├── chat-service/                    # Conversation logic, context management (Go)
│   └── internal/
│       ├── domain/                  # Entities, repository interfaces
│       ├── application/             # Use cases
│       └── infrastructure/
│           ├── context/             # ContextBuilder, Budget, Compressor, TopicAnalyzer
│           ├── mq/                  # RocketMQ producer/consumer
│           ├── persistence/         # Redis + PostgreSQL (GORM)
│           └── tokenizer/           # tiktoken-go
├── llm-inference/                   # Inference engine (Python)
│   └── src/optimization/
│       ├── kv_cache.py              # KV cache + prefix cache
│       └── speculative_decoding.py  # Draft-target verification
├── finetune/                        # LoRA/QLoRA (110 tests)
├── alignment/                       # DPO (50 tests)
├── rlhf/                            # PPO RLHF (21 tests)
├── evaluation/                      # MMLU/C-Eval/GSM8K/HumanEval (90 tests)
├── rag/                             # RAG pipeline (51 tests)
└── synthetic-data/                  # Self-instruct, EDA (38 tests)

inference-engine/                    # Optimization experiments
├── design.md
├── memory-manager/
│   ├── kv_cache_manager.py          # BlockPool + eviction policies + PrefixCache
│   └── engine_cache_adapter.py      # Engine injection adapter
├── scheduler/
│   └── continuous_batching.py       # Iteration-level scheduling
├── benchmark/
│   ├── latency_bench.py             # TTFT/TPOT estimation
│   ├── throughput_bench.py          # Tokens/sec vs concurrency
│   ├── memory_bench.py              # Memory scaling analysis
│   ├── quality_bench.py             # Accuracy impact reference
│   └── quantization_pipeline.py     # GPU/CI dual-mode benchmark
└── tests/                           # 68 tests
```

---

## Quick Start

```bash
git clone https://github.com/Einspanner123/free-chat.git
cd free-chat
cp .env.example .env
docker compose up -d --build
```

Run benchmarks:
```bash
python3 inference-engine/benchmark/quantization_pipeline.py
python3 inference-engine/scheduler/continuous_batching.py
python3 services/experiments/bench_inference.py
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
| inference-engine | 68 | KV cache, scheduler, benchmarks |
| llm-inference | 145 | Engine backends, quantization, optimizations |
| finetune | 110 | LoRA/QLoRA training, data loading, merging |
| evaluation | 90 | MMLU/C-Eval/GSM8K/HumanEval execution |
| rag | 51 | Retrieval strategies, chunking |
| alignment | 50 | DPO loss, preference data |
| synthetic-data | 38 | Generation, quality filtering |
| rlhf | 21 | PPO loss, GAE estimation |

Total: **573 tests**.
