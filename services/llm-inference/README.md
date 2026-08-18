# llm-inference

gRPC inference service with pluggable engines (**HF** transformers fallback and
**vLLM** high-throughput serving) and quantization support.

## Quick start

```bash
# HF engine (fallback)
ENGINE_TYPE=hf MODEL_NAME=Qwen/Qwen3-0.6B .venv/bin/python src/server.py

# vLLM engine (default)
ENGINE_TYPE=vllm MODEL_NAME=Qwen/Qwen3-0.6B .venv/bin/python src/server.py
```

### Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `ENGINE_TYPE` | `vllm` | `vllm`, `hf`, or `auto` |
| `MODEL_NAME` | `Qwen/Qwen3-0.6B` | Target model HF id / path |
| `MAX_TOKENS` | `512` | Max generated tokens |
| `TEMPERATURE` / `TOP_P` / `TOP_K` / `REPETITION_PENALTY` | `0.7` / `0.8` / `40` / `1.1` | Sampling params |
| `GRPC_PORT` | `8083` | gRPC listen port |
| `DRAFT_MODEL` | *(unset)* | Draft model HF id → enables speculative decoding |
| `SPECULATIVE_GAMMA` | `5` | Draft tokens proposed per verify round |
| `SPECULATIVE_ENABLED` | `true` | Secondary gate (`DRAFT_MODEL` is the master switch) |
| `KV_EVICTION_WINDOW` | `0` (off) | Recent KV tokens kept per layer → enables KV eviction |
| `KV_EVICTION_SINK` | `4` | Attention-sink tokens always kept (recommend ≥ 4) |

## Speculative decoding

Enabled by setting `DRAFT_MODEL`. Two backends, both real:

- **vLLM engine** — uses vLLM's native `draft_model` speculative decoding
  (`spec_model` / `spec_tokens` in `AsyncEngineArgs`). This is the
  production serving path.
- **HF engine** — runs the real Leviathan et al. (2023) draft-verify loop
  (`src/optimization/speculative_decoding.py`): incremental KV-cache reuse,
  one parallel verification forward per round, rejection sampling, and KV
  `crop()` rollback on rejection. Both draft and target are warped with the
  same logits processors (repetition penalty → temperature → top_k → top_p),
  so the output distribution matches the non-speculative `do_sample=True` path.

The implementation is the service-layer home of the algorithm benchmarked in
`research/inference_optimization/run_speculative_real.py`.

**Requirements**

- The draft model must share the target's tokenizer vocab (token ids map 1:1),
  validated at decoder construction (probe-string id equality; a different
  added-token tail, e.g. Qwen2.5 draft + Qwen3 target, is fine).
- `temperature > 0` (greedy sampling is unsupported in the spec loop).
- **vLLM backend: lower `GPU_MEMORY_UTILIZATION` when a draft is set.** Both
  the target and draft models must fit inside the reserved pool. The default
  `0.9` fails whenever the GPU is less than ~90% free (vLLM raises
  `ValueError: Free memory ... less than desired GPU memory utilization` at
  `init_device` — this is a memory-headroom error, not a speculative-decoding
  one). With a small target + small draft, `0.6` is plenty:

```bash
GPU_MEMORY_UTILIZATION=0.6 \
ENGINE_TYPE=vllm MODEL_NAME=Qwen/Qwen3-0.6B DRAFT_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  .venv/bin/python src/server.py
```

```bash
# HF engine + speculative decoding (Qwen2.5-0.5B draft for Qwen3-0.6B)
ENGINE_TYPE=hf \
  MODEL_NAME=Qwen/Qwen3-0.6B \
  DRAFT_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  SPECULATIVE_GAMMA=5 \
  .venv/bin/python src/server.py

# vLLM engine + native speculative decoding
ENGINE_TYPE=vllm \
  MODEL_NAME=Qwen/Qwen3-0.6B \
  DRAFT_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  .venv/bin/python src/server.py
```

## KV-cache eviction (StreamingLLM-style)

Enabled by setting `KV_EVICTION_WINDOW`. The HF engine passes a
**`SinkWindowCache`** (`src/optimization/kv_eviction.py`) as
`past_key_values` to `generate()`: during each single-token decode step it
keeps only `[sink tokens | recent window]` of K/V per layer, evicting the
middle of the context.

**The goal is MEMORY, not speed** — the earlier `kv_cache_speedup.json`
showed eviction does not speed up decode. It lets a fixed GPU hold a far
longer context / larger batch. The benchmark
`research/inference_optimization/run_kv_eviction_quality.py` measures the
memory × quality curve (decode-phase peak, exact `kv_bytes()`, NIAH
recall-by-position, perplexity).

```bash
ENGINE_TYPE=hf \
  MODEL_NAME=Qwen/Qwen3-0.6B \
  KV_EVICTION_WINDOW=512 \
  KV_EVICTION_SINK=4 \
  .venv/bin/python src/server.py
```

**Requirements & limitations**

- **Mutually exclusive with speculative decoding** (`DRAFT_MODEL`): the
  speculative loop rolls back the KV cache with `crop()`, which
  `SinkWindowCache` forbids (its layout is `[sink|window]`, not a prefix).
  Setting both raises at config time.
- Single sequence only (no padding/packing); prefill-phase peak memory is NOT
  saved (eviction starts on the first decode step) — only decode steady-state
  memory is saved.
- Not compatible with `cache_implementation` (transformers raises when both a
  cache instance and `cache_implementation` are passed).

## Tests

```bash
# GPU-free unit tests (spec decoder uses tiny fake models)
../../.venv/bin/python -m pytest tests -q

# Real-model GPU smoke test (opt-in; Qwen2.5-0.5B draft + Qwen3-0.6B target)
SPEC_E2E=1 ../../.venv/bin/python -m pytest tests/test_speculative_decoding.py -m gpu -v
```
