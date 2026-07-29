# Free Chat — LLM Application Platform with Inference Optimization Experiments

<a href="https://github.com/Einspanner123/free-chat"><img src="https://img.shields.io/badge/GitHub-Free%20Chat-blue?logo=github"></a>

A microservices-based LLM application platform covering chat serving, model fine-tuning, preference alignment, RAG, and evaluation. The platform includes a set of inference optimization experiments built on top of the serving stack.

---

## Project Overview

Free Chat is organized in two layers:

```
                    Chat Application Layer
    ┌──────────────────────────────────────────────┐
    │  API Gateway (Go)   │   Auth Service (Go)    │
    │  Chat Service (Go)  │   Web UI               │
    │  PostgreSQL / Redis  │   RocketMQ             │
    └───────────────────────┬──────────────────────┘
                            │ gRPC
    ┌───────────────────────▼──────────────────────┐
    │  LLM Lifecycle Layer                         │
    │  Inference · Fine-tuning · Alignment · RAG   │
    │  Evaluation · Data Synthesis                 │
    └──────────────────────────────────────────────┘
```

The **chat application layer** (Go) handles user auth, session management, conversation logic, and API routing. The **LLM lifecycle layer** (Python) provides inference serving, model customization, and evaluation tooling.

---

## Inference Optimization Experiments

The following optimization modules live alongside the application code, each with its own benchmark data.

### 1. KV Cache Manager

**File**: `services/llm-inference/src/optimization/kv_cache.py`

Implements KV cache management with allocation, eviction, and prefix reuse.

| Method | Description |
|--------|-------------|
| KVCache | LRU-evicted key-value store for KV tensors across requests |
| PrefixCache | Stores KV states keyed by prompt hash; on match, reuses precomputed states |

**Benchmark** (simulated): For a 7B model at 32K sequence length, the full KV cache occupies ~16GB. With LRU eviction at 50% memory budget, accuracy retention is ~98%.

```
Memory scaling (FP16):
  Seq Len → 1K    4K     8K     32K    128K
  0.5B model:  0.1GB  0.2GB  0.4GB  1.7GB   6.7GB
  7B model:    0.5GB  2.0GB  4.0GB  16.0GB  64.0GB
  70B model:   2.6GB  10.5GB 21.0GB 84.0GB  335.5GB
```

### 2. Continuous Batching Scheduler

**File**: `inference-engine/scheduler/continuous_batching.py`

Implements iteration-level scheduling (Orca, SOSP 2022) where requests enter and leave the batch at each decoding step, rather than waiting for full-batch completion.

```
Static batching:        [A B C D] all wait for longest → 46.85s for 32 reqs
Continuous batching:    A B C → A B D → A E D → ... → 0.25s for 32 reqs
```

| Method | 32 Reqs Time | Throughput | Avg Latency | P99 Latency |
|--------|-------------|------------|-------------|-------------|
| Static | 46.85s | 101 t/s | 11.71s | 12.70s |
| Continuous | 0.25s | 19,064 t/s | 0.18s | 0.25s |

Simulated on 7B-scale model, 50ms/token, mixed request lengths (32–256 tokens).

### 3. Quantization Support & Analysis

**File**: `services/llm-inference/src/quantization.py`

Supports AWQ, GPTQ, and SqueezeLLM quantization, selectable via the `QUANTIZATION` environment variable.

Reference benchmark for Qwen2.5-7B (published numbers):

| Method | VRAM | Latency | MMLU |
|--------|------|---------|------|
| FP16 | 14.0 GB | 45 ms/t | 70.1% |
| AWQ INT4 | 5.0 GB | 32 ms/t | 69.5% |
| GPTQ INT4 | 5.5 GB | 35 ms/t | 68.8% |

### 4. Speculative Decoding

**File**: `services/llm-inference/src/optimization/speculative_decoding.py`

Draft-verify loop: a small model generates γ candidate tokens, the target model verifies them in one forward pass.

Speedup formula: `1 / (1 - α + α/γ)`, where α is token acceptance rate and γ is draft length. At α = 0.8 and γ = 5, theoretical speedup is 2.78×.

---

## LLM Lifecycle Modules

### Fine-tuning (LoRA / QLoRA)

**Directory**: `services/finetune/`

Parameter-efficient fine-tuning with configurable rank, target modules, and quantization.

### Preference Alignment (DPO)

**Directory**: `services/alignment/`

Direct Preference Optimization as an alternative to RLHF.

### RAG Pipeline

**Directory**: `services/rag/`

Document chunking, dense/sparse/hybrid retrieval, and generation.

### Evaluation Suite

**Directory**: `services/evaluation/`

Model evaluation on MMLU (57 subjects), C-Eval (Chinese), GSM8K (math), and HumanEval (code).

### Synthetic Data

**Directory**: `services/synthetic-data/`

Self-instruct, evol-question, and EDA-based data generation.

---

## Project Structure

```
inference-engine/                      # Inference optimization experiments
├── design.md                          # Full design document
├── scheduler/
│   └── continuous_batching.py         # Iteration-level scheduling + KVCache integration
├── memory-manager/
│   ├── kv_cache_manager.py            # BlockPool + 3 eviction policies + PrefixCache
│   └── engine_cache_adapter.py        # Adapter for engine injection
├── benchmark/
│   ├── benchmark_runner.py            # BenchResult, BenchmarkSuite, output formats
│   ├── latency_bench.py               # TTFT, TPOT estimation
│   ├── throughput_bench.py            # Tokens/sec under concurrency
│   ├── memory_bench.py                # Memory scaling + GPU fit analysis
│   ├── quality_bench.py               # Accuracy impact reference
│   ├── kv_cache_profiling.py          # KV cache memory scaling
│   ├── quantization_bench.py          # Quantization comparison tables
│   └── quantization_pipeline.py       # GPU/CI dual-mode benchmark
└── tests/                             # 68 tests

services/
├── api-gateway/                     # HTTP gateway (Go)
├── auth-service/                    # User authentication (Go)
├── chat-service/                    # Conversation logic (Go)
├── llm-inference/                   # Inference engine (Python)
│   └── src/optimization/
│       ├── kv_cache.py              # KV cache + prefix cache
│       └── speculative_decoding.py  # Draft-target verification
├── finetune/                        # LoRA/QLoRA
├── alignment/                       # DPO
├── rlhf/                            # PPO RLHF
├── evaluation/                      # MMLU/C-Eval/GSM8K/HumanEval
├── rag/                             # RAG pipeline
└── synthetic-data/                  # Data generation

services/experiments/                # Benchmark runner scripts
├── bench_inference.py               # Engine latency/throughput
└── bench_finetune.py                # Fine-tuning ablation
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
# Continuous batching + KV cache integration demo
python3 inference-engine/scheduler/continuous_batching.py

# KV cache memory profiling across model sizes
python3 inference-engine/benchmark/kv_cache_profiling.py

# Quantization comparison tables
python3 inference-engine/benchmark/quantization_bench.py

# Quantization pipeline (CI mode with reference data)
python3 inference-engine/benchmark/quantization_pipeline.py

# Add --gpu to run on real hardware:
# python3 inference-engine/benchmark/quantization_pipeline.py --gpu --model Qwen/Qwen2.5-7B

# Latency, throughput, memory, quality benchmarks
python3 services/experiments/bench_inference.py
python3 services/experiments/bench_finetune.py
```

Run all inference-engine tests:
```bash
python3 -m pytest inference-engine/tests/
```

---

## Test Coverage

| Module | Tests | Scope |
|--------|-------|-------|
| inference-engine (optimizations) | 68 | KV cache, scheduler, benchmarks, quantization pipeline |
| llm-inference | 145 | Engine switching, KV cache, speculative decoding, quantization |
| finetune | 110 | LoRA/QLoRA training, data format merging |
| evaluation | 90 | MMLU/C-Eval/GSM8K/HumanEval, metrics |
| rag | 51 | Retrieval strategies, chunking |
| alignment | 50 | DPO loss, preference data |
| synthetic-data | 38 | Generation, filtering, EDA |
| rlhf | 21 | PPO loss, GAE estimation |

Total: **573 tests**.
