# Claims ledger

Only entries marked `VERIFIED` may appear as performance or production claims
in project-facing material.

| Claim ID | Status | Claim | Evidence |
|---|---|---|---|
| architecture-contract | VERIFIED | The repository defines versioned Agent hints, capability-first placement and lifecycle-aware KV policy. | Unit and scale tests in this repository. |
| harness-contracts | VERIFIED | Four named Harness adapters emit the same lifecycle and identity depth. | `libs/harness-adapters/tests/test_adapters.py`. |
| control-images | VERIFIED | Gateway and WebUI images build from locked base-image digests and start as non-root services. | workstation container build and in-container health checks, 2026-08-31. |
| control-state | VERIFIED | Worker registration, route leases and decision explanations survive a scheduler restart through etcd; lifecycle events receive JetStream acknowledgements and deterministic message-ID deduplication. | Digest-pinned Compose integration on workstation, 2026-08-31; `services/scheduler/src/freechat_scheduler/live_validation.py`; `services/trace-replay/tests/test_bus.py`. This does not establish multi-replica HA. |
| engine-hook-contract | VERIFIED | The pinned vLLM fork exposes opt-in allocate, hit, free and eviction callbacks; authenticated tenant/cache identity and generation fences are preserved by the worker bridge without blocking the engine callback thread. | Fork commit `1c191f8632f1bfdd57d5e1de3b749b60f824f98a`; 94 focused fork tests and `worker/tests/test_vllm_adapter.py`. This does not establish a running model-server integration. |
| kv-quantize-microbenchmark | VERIFIED | Fused per-group FP16 KV int8 quantization reduces isolated kernel latency for the recorded shape on ross A6000 and workstation A5000/A4000 under the locked stack. | `docs/kernel-candidates.md`; raw artifact SHAs `2253071d9d17c0b231aaa6156c0b6444cb9fc085a924f05ab33708c1bd87b1b8`, `c44f7eb393e0b7f3607c63f170f2e143d49f7ffdb8ed356dd0ea01b77dbbe735`, and `a506160491b2171c03465d1f2c9c10a41c0a35c9a8b7fd579bd662d9f613d746`. This is not an end-to-end claim. |
| gpu-performance | UNVERIFIED | GPU or Agent-task performance improvement. | Requires target-system benchmark artifacts. |
| production-ha | BLOCKED | Production HA validation. | Requires a third physical failure domain. |
| kernel-e2e | UNVERIFIED | Triton or memory-pipeline end-to-end gain. | Requires correctness, profiler, microbenchmark and Agent trace evidence. |
