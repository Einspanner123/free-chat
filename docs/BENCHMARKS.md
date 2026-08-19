# Benchmarks — Audit-Ready Reference

This document maps every headline number in the README/README_CN to the **exact script that produced it** and the **on-disk result file** that stores it. It also states the **baseline design** honestly so the comparison structure can be scrutinized.

> The `research/**/results/*.json` files are git-ignored (regenerable by re-running the scripts). This doc is the auditable index: clone the repo, run a script, and verify the number reproduces.

---

## 1. Baseline Design (read this first)

The framework compares four kinds of baselines. They are NOT interchangeable, and each headline number belongs to exactly one category:

| Baseline | What it is | Role | Fairness |
|---|---|---|---|
| **Truncation** | Take the first N tokens of the document (budget 512/1024/2048/4096) | **Floor** — "what happens with no method at all" | Weak as a *competing method*; valid as a lower bound. The answer span is usually outside the first N tokens, so 10% is expected. |
| **Full Context** | Feed the whole document (up to model window) | **Upper reference** — the model's best-case when budget is unlimited | A 0.6B model degrades past ~8K tokens, so Full Context is often *worse* than truncation on small models (lost-in-the-middle). |
| **Same-budget strategies** (project_topic / attention_sink / sink_topic) | Compress+layout the SAME N tokens differently | **Method vs. method** — the fair comparison. Both sides use identical token budget; only the *selection/composition* differs. | This is where the compress/layout layers earn their claim. |
| **BM25 retrieval** (top-1/top-k) | Retrieve the relevant passage(s), stuff into budget | **Retrieval baseline** — isolates the retrieval stage's contribution | A different axis: retrieval *finds* the needle; compression *fits* the budget. The auto-router combines both. |

**Key honesty point**: the "10% → 98%" headline on passage retrieval is a **floor-vs-retrieval** comparison (truncation vs BM25). It is NOT a claim that compression/layout alone reaches 98%. The compression/layout stages' *own* contribution over truncation on passage retrieval is **6.5×–9.5×** (10% → 65–95%), still strong, and reported below — but the 98% is BM25's job, and the auto-router's win is *selecting* BM25 for this task type.

---

## 2. Long-Context Enhancement (retrieve → compress → layout)

### 2.1 Passage localization — LongBench `passage_retrieval_en`

| Model | Strategy | Budget | Accuracy | × over truncation | Source |
|---|---|---|---|---|---|
| Qwen3-0.6B | truncation | 1024 | **8–10%** | 1.0× (floor) | `passage_retrieval_en_Qwen3-0.6B_auto.json`, `passage_retrieval_qwen3_06b.json` |
| Qwen3-0.6B | project_topic (compress+layout) | 1024 | **65%** | **6.5×** | `passage_retrieval_qwen3_06b.json` |
| Qwen3-0.6B | attention_sink | 1024 | 60% | 6.0× | `passage_retrieval_qwen3_06b.json` |
| Qwen3-0.6B | sink_topic | 1024 | 60% | 6.0× | `passage_retrieval_qwen3_06b.json` |
| Qwen3-0.6B | **bm25_top1** | 1024 | **98%** | **9.8×** | `passage_retrieval_en_Qwen3-0.6B_auto.json` |
| Qwen3-0.6B | **auto (router → bm25)** | 1024 | **98%** | 9.8× | `passage_retrieval_en_Qwen3-0.6B_auto.json` |
| Qwen2.5-7B | truncation | 1024 | 10% | 1.0× | `passage_retrieval_qwen25_7b_20.json` |
| Qwen2.5-7B | project_topic | 1024 | **95%** | **9.5×** | `passage_retrieval_qwen25_7b_20.json` |
| Qwen2.5-7B | attention_sink / sink_topic | 1024 | **100%** | 10× | `passage_retrieval_qwen25_7b_20.json` |

- **Script**: `research/longbench_v1/run_passage_retrieval.py`, `run_passage_retrieval_auto.py`
- **Cross-scale**: 0.6B 7.4× / 7B 10× over truncation (the 7B exploits layout better — strategy value grows with model capability).
- **Auto-router**: classifies all 50/50 queries as `locate`, routes to `bm25_top1`, reaches 98% — no manual strategy selection.
- **Pitfall caught**: an answer hint like `(e.g., Paragraph 5)` anchors the 0.6B to output "5" (75% → 95% once removed); empty BM25 result now falls back to recency truncation.

### 2.2 Single-doc QA — LongBench `multifieldqa_en` (same-budget, fair)

| Model | Strategy | Budget | F1 | × over truncation | Source |
|---|---|---|---|---|---|
| Qwen3-0.6B | truncation | 512 | 0.174 | 1.0× | `multifieldqa_en.json` |
| Qwen3-0.6B | **project_topic** | 512 | **0.357** | **2.05×** | `multifieldqa_en.json` |
| Qwen3-0.6B | attention_sink | 512 | 0.321 | 1.84× | `multifieldqa_en.json` |
| Qwen3-0.6B | truncation | 1024 | 0.191 | 1.0× | `multifieldqa_en.json` |
| Qwen3-0.6B | project_topic | 1024 | 0.332 | 1.74× | `multifieldqa_en.json` |

- **Script**: `research/longbench_v1/run_all_tasks.py`
- This is the **fairest** comparison: identical 512/1024 token budget, only selection differs. Compression+layout genuinely wins 2.1× at the low budget where every token matters.

### 2.3 Science QA — LongBench `qasper` (same-budget, fair)

| Model | Strategy | Budget | F1 | × over truncation | Source |
|---|---|---|---|---|---|
| Qwen3-0.6B | truncation | 1024 | 0.132 | 1.0× | `qasper.json` |
| Qwen3-0.6B | **project_topic** | 1024 | **0.253** | **1.9×** | `qasper.json` |
| Qwen3-0.6B | attention_sink | 1024 | 0.218 | 1.65× | `qasper.json` |

- **Script**: `research/longbench_v1/run_all_tasks.py`

### 2.4 Task boundary (negative results, honestly reported)

| Task | Finding | Source |
|---|---|---|
| narrativeqa (Qwen3-0.6B, 31K ctx) | full_context F1=0.146, truncation_8k F1=0.146, **bm25_top3 F1=0.127 (BM25 HURTS)** — narrative answers need synthesis, not localization; retrieval focuses but generation is model-limited | `narrativeqa_boundary.json` |
| passage_count | all strategies 2.5% — the task requires counting, not locating; framework boundary | `passage_count.json` |
| trec (0.6B) | project_topic 6.7%, attention_sink 3.3%, truncation 0% — gains exist but absolute accuracy is low | `trec.json` |

- **Boundary conclusion**: the framework helps **localization** (find the needle) and **budget-constrained QA** (compress to fit); it does NOT help **generation-heavy / counting** tasks where the bottleneck is the model itself.

---

## 3. Inference & KV Cache Optimization (RTX A6000)

### 3.1 Batch throughput

| Batch | tokens/s | × vs batch-1 | Source |
|---|---|---|---|
| 1 | 25.7 | 1.0× | `decode_optimization.json` |
| 2 | 46.8 | 1.82× | `decode_optimization.json` |
| 4 | (see file) | ~3.5× | `decode_optimization.json` |
| **8** | (see file) | **6.23×** | `decode_optimization.json` |

- **Script**: `research/inference_optimization/run_decode_optimization.py`
- **Negative result**: bitsandbytes INT8 is **5.7× SLOWER** (dequantization overhead); INT8 value is memory reduction, not speed on A6000.

### 3.2 Prefix cache prefill speedup

| Suffix length | Full prefill (ms) | Cached prefill (ms) | Speedup | Source |
|---|---|---|---|---|
| 5 tokens | 124.2 | 41.8 | **2.97×** | `kv_cache_speedup.json` |
| 22 tokens | 124.7 | 44.9 | ~2.78× | `kv_cache_speedup.json` |
| (other configs) | — | — | 1.68–2.97× | `kv_cache_speedup.json` |

- **Script**: `research/inference_optimization/run_kv_cache_speedup.py`
- **Reported as "up to 2.97×"** (range 1.68–2.97×); the 2.97× is the short-suffix best case.

### 3.3 KV eviction — memory × quality

| Window | Keep tokens | KV bytes | Context multiplier | Recall | Source |
|---|---|---|---|---|---|
| baseline (no eviction) | 16384 | 1.88 GB | 1.0× | 1.0 | `kv_eviction_quality.json` |
| **256** | 260 | ~30 MB | **63×** | (see file) | `kv_eviction_quality.json` |
| 512 | 516 | — | 31× | (see file) | `kv_eviction_quality.json` |

- **Script**: `research/inference_optimization/run_kv_eviction_quality.py`
- **Headline**: 1794 MB → 28 MB = **63× same-card context capacity** (window=256, sink=4).

### 3.4 KV low-rank analysis (MLA analogy)

| Layer | head_dim | rank95 (token redundancy) | PC-95 (per-token dim redundancy) | Source |
|---|---|---|---|---|
| layer 0 | 128 | **2.0** | 10.4 | `kv_lowrank.json` |
| layer 14 | 128 | 49.5 | 69.1 | `kv_lowrank.json` |

- **Script**: `research/inference_optimization/run_kv_lowrank.py`
- **Finding**: early-layer KV is extremely low-rank (rank95≈2 of 128) → supports aggressive latent compression; deep layers are near-full-rank (rank95≈50).
- **MLA recon error** (PCA latent dim → attention output error): latent=8 → 8.8%, latent=16 → 4.8%, latent=32 → 2.6%, **latent=64 → 0.85%**. This is why the serve-time `CompressedKVCache` default targets latent=64.

### 3.5 RoPE length extension

| Mode | 30K | 60K | 80K | Source |
|---|---|---|---|---|
| default (NTK-aware) | 1.0 | 1.0 | 1.0 | `ntk_extension.json` |
| YaRN (factor 2.0) | 1.0 | 1.0 | **0.75** | `ntk_extension.json` |

- **Script**: `research/inference_optimization/run_ntk_extension.py`
- **Negative result**: YaRN gives **no gain** at 30K/60K and **degrades** at 80K (0.75) — NTK default already handles 80K on this model.

### 3.6 Speculative decoding (draft-verify)

| Metric | Value | Source |
|---|---|---|
| mean accept rate | 0.172 | `speculative_real.json` |
| mean speedup | **0.26× (SLOWER)** | `speculative_real.json` |
| expected tokens/verify | 1.21 | `speculative_real.json` |

- **Script**: `research/inference_optimization/run_speculative_real.py`
- **Negative result**: draft=Qwen2.5-0.5B, target=Qwen2.5-7B, gamma=5. Accept rate too low (17%) → 0.26× (slower). Draft quality must approach target for speculative to pay off.

---

## 4. Serve-time features (production flags, off by default)

| Feature | Env flag | Default | Validated by | Commit |
|---|---|---|---|---|
| Tail latency p50/p95/p99 | (always on for benchmarks) | on | `test_latency_stats.py` (CPU) | `7e0306f` |
| Prefix KV reuse | `PREFIX_CACHE_ENABLED` | off | `test_prefix_cache.py` (CPU) + `test_prefix_reuse_e2e.py` (GPU, gated `KV_E2E=1`) | `715acd0`, `fa283a8` |
| MLA latent KV compression | `KV_COMPRESSION=mla` | `none` | `test_kv_compression.py` (CPU) + `test_real_mla_generate_matches_baseline` (GPU, gated) | `cdd990e`, `fa283a8` |

- **E2E verified on transformers 5.14.1, Qwen3-0.6B**: prefix reuse (miss/hit/partial-hit) matches greedy baseline token-for-token; MLA latent=64 matches baseline, latent=16 diverges (quality cliff, documented).

---

## 5. How to reproduce

```bash
# Long-context benchmarks (CPU OK for small models; GPU for 7B)
cd research/longbench_v1 && python run_passage_retrieval.py
cd research/longbench_v1 && python run_all_tasks.py

# Inference optimization (GPU required)
cd research/inference_optimization && python run_decode_optimization.py
cd research/inference_optimization && python run_kv_cache_speedup.py
cd research/inference_optimization && python run_kv_eviction_quality.py
cd research/inference_optimization && python run_kv_lowrank.py
cd research/inference_optimization && python run_ntk_extension.py
```

Each script writes to `research/*/results/<name>.json`. Results are git-ignored and regenerated by the scripts above.
