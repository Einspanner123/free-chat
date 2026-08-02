[English](README.md) | [中文](README_CN.md)

# Free Chat -- Long-Context Framework for Small Models

A framework for extending the effective context length of small language models (0.5B-3B) through context management and RAG-based retrieval. Go control plane, Python compute plane.

---

## Overview

The project has two layers:

- **Application layer** (`services/`): runnable microservices (chat, auth, gateway, inference, RAG, fine-tuning, evaluation).
- **Research layer** (`research/`): benchmarks and experiments validating the context framework on real hardware.

The core work is the **context-engine**: a layered pipeline (retrieval → compression → layout) that prepares optimized contexts for small models under a token budget.

---

## Project Structure

```
services/                    # Application layer
├── api-gateway/             # HTTP gateway (Go)
├── auth-service/            # User auth (Go)
├── chat-service/            # Chat service with context management (Go)
│   └── internal/interfaces/context_client.go  # gRPC client to context-engine
├── llm-inference/           # Inference engine: HF/vLLM backends (Python)
├── context-engine/          # Context optimization: strategies/retriever/pipeline + gRPC server
├── rag/                     # RAG: chunking, embedding, BM25/dense/hybrid retrieval
├── finetune/                # LoRA/QLoRA fine-tuning
├── alignment/               # DPO preference alignment
├── rlhf/                    # PPO RLHF
├── evaluation/              # MMLU, C-Eval, GSM8K, HumanEval
└── synthetic-data/          # Self-instruct, data augmentation

research/                    # Research layer
├── long_context/            # Needle-in-a-haystack, compression ablations
├── longbench_v1/            # LongBench multi-task evaluation
├── longbench/               # LongBench-style QA (v2)
├── loong/                   # Chinese multi-doc QA
├── zero_scrolls/            # Long-text comprehension
└── inference_optimization/  # Real inference measurements

pkg/proto/contextengine/     # Shared gRPC contract (Go + Python stubs)
scripts/download_benchmark_data.py  # Fetch benchmark datasets on demand
```

---

## Context-Engine Design

The context-engine (`services/context-engine/`) builds optimized contexts under a token budget. It has three layers:

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│  Retriever  │ → │  Compressor  │ → │   Layout    │
│ BM25/Dense  │   │  tiered      │   │ sink/topic  │
│ keyword     │   │  truncation  │   │             │
└─────────────┘   └──────────────┘   └─────────────┘
```

| Layer | File | Responsibility |
|-------|------|----------------|
| strategies | `strategies.py` | Stateless primitives: chunking, keyword extraction, truncation, relevance selection, tiered compression, attention-sink layout |
| retriever | `retriever.py` | Common interface + factory: BM25 (pure Python), keyword, dense (optional embedding) |
| pipeline | `pipeline.py` | Orchestration: retrieve → compress → layout → assemble |

The engine is exposed as a gRPC service (`grpc_server.py`), and the Go chat-service calls it through `ContextClient` implementing `domain.ContextOptimizer`.

---

## Application / Research Boundary

The boundary is explicit:

| Layer | Purpose | Data | Stability |
|-------|---------|------|-----------|
| `services/` | Production features | No external datasets | Tested (559+ tests) |
| `research/` | Experiments, benchmarks, findings | Large datasets (gitignored) | Exploratory |
| `pkg/proto/` | Shared contracts | - | Stable interface |

Benchmark datasets (495MB) are not committed to git. They are fetched via `scripts/download_benchmark_data.py`.

---

## Key Findings

Experiments run on NVIDIA RTX A6000 with Qwen3-0.6B and Qwen2.5-7B.

### LongBench passage_retrieval_en

Task: given a multi-paragraph document, find the paragraph matching a description. 200 samples, ~12.7K tokens each.

| Method | Accuracy |
|--------|----------|
| Truncation | 10% |
| Keyword compression | 74% |
| **BM25 retrieval (top-1)** | **98%** |

BM25 retrieval hit rate is 100% (answer paragraph always in top-1). The 0.6B model achieves 98% accuracy with a single retrieved paragraph.

### Model-scale invariance

Same compressed context, two model sizes (20 samples):

| Strategy | Qwen3-0.6B | Qwen2.5-7B |
|----------|-----------|------------|
| Truncation | 10% | 10% |
| Project + Topic | 74% | 95% |
| Attention Sink | 60% | 100% |
| Sink + Topic | 60% | 100% |

Framework gain is scale-invariant: 7.4x (0.6B) and 10x (7B) over truncation. Strategy value grows with model capability (7B exploits layout better).

### Task boundary

| Task type | Framework effect |
|-----------|-----------------|
| Passage-location (passage_retrieval_en) | 98-100% (killer feature) |
| Single-doc QA (multifieldqa_en) | F1 0.174 → 0.357 (2.1x) |
| Science QA (qasper) | F1 0.132 → 0.253 (1.9x) |
| Narrative generation (narrativeqa) | No gain (answer synthesis, not location) |
| Chinese comprehension / classification | Limited (0.6B comprehension boundary) |

---

## Test Coverage

| Module | Tests |
|--------|-------|
| context-engine | 47 (strategies, retriever, pipeline, gRPC) |
| llm-inference | 153 |
| finetune | 115 |
| evaluation | 90 |
| rag | 52 |
| alignment | 50 |
| synthetic-data | 38 |
| rlhf | 21 |
| chat-service (Go) | + context client tests |
| long_context research | 14 |

---

## Quick Start

```bash
# 1. Install dependencies
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
pip install torch transformers

# 2. Run chat platform
cp .env.example .env
docker compose up -d --build

# 3. Start context-engine gRPC service
.venv/bin/python -m grpc_server --port 8089

# 4. Run benchmarks (download data first)
python scripts/download_benchmark_data.py
.venv/bin/python research/longbench_v1/run_passage_retrieval.py
```

---

## Repository Size

Benchmark data is excluded from git. The repository is ~51MB of source code and generated artifacts, with data fetched on demand.
