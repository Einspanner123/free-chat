# Claims ledger

Only entries marked `VERIFIED` may appear as performance or production claims
in project-facing material.

| Claim ID | Status | Claim | Evidence |
|---|---|---|---|
| architecture-contract | VERIFIED | The repository defines versioned Agent hints, capability-first placement and lifecycle-aware KV policy. | Unit and scale tests in this repository. |
| harness-contracts | VERIFIED | Four named Harness adapters emit the same lifecycle and identity depth. | `libs/harness-adapters/tests/test_adapters.py`. |
| control-images | VERIFIED | Gateway and WebUI images build from locked base-image digests and start as non-root services. | workstation container build and in-container health checks, 2026-08-31. |
| control-state | VERIFIED | Worker registration, route leases and decision explanations survive a scheduler restart through etcd; lifecycle events receive JetStream acknowledgements and deterministic message-ID deduplication. | Digest-pinned Compose integration on workstation, 2026-08-31; `services/scheduler/src/freechat_scheduler/live_validation.py`; `services/trace-replay/tests/test_bus.py`. This does not establish multi-replica HA. |
| engine-hook-contract | VERIFIED | The pinned vLLM fork exposes opt-in allocate, hit, free and eviction callbacks; authenticated tenant/cache identity and generation fences are preserved by the worker bridge without blocking the engine callback thread. | Fork commit `5871f38ee947760ed7f5dd5487bb04456db8bc98`; 97 focused fork tests, `worker/tests/test_vllm_adapter.py`, and an A4000 serving probe that correlated HTTP lifecycle identity with EngineCore allocate/free/hit events. The serving probe used dummy weights and is not performance evidence. |
| lifecycle-retention-contract | VERIFIED | The opt-in vLLM policy keeps bounded, high-priority, near-resume blocks behind ordinary LRU eviction candidates and releases protection on resume, terminal state, cancellation, eviction or expiry; internal Chat Completions, Completions and Responses requests carry authenticated lifecycle metadata into `SamplingParams`. | Fork commit `5871f38ee947760ed7f5dd5487bb04456db8bc98`; bounded-capacity, ordering, expiry, resume and three-protocol tests. This is contract evidence, not a cache-hit or end-to-end performance claim. |
| lifecycle-retention-mechanism | MECHANISM_ONLY | Under an intentionally constrained 32 MiB cache, the serving policy changed eviction order and preserved the target resume prefix across deterministic pressure. | A4000 dummy-weight probe, 2026-09-01: native policy retained 16 of 930 prompt tokens; lifecycle policy retained 928. Raw baseline/candidate event hashes are recorded in `docs/environment-validation.md`. This single probe is forbidden in README/resume performance claims and does not replace real-model paired Harness trials. |
| kv-quantize-historical | HISTORICAL | The isolated fused KV INT8 kernel measured 6.2–7.7x on three GPUs under the superseded PyTorch 2.11/Triton 3.6 evidence stack. | `docs/kernel-candidates.md`; retained for provenance only and forbidden in current resume material until replicated under the current lock. |
| kv-quantize-current | UNVERIFIED | Fused KV INT8 kernel improvement under the current deployable worker stack. | Requires three-GPU correctness, profiler, microbenchmark and raw artifacts under PyTorch 2.13/Triton 3.7. |
| gpu-performance | UNVERIFIED | GPU or Agent-task performance improvement. | Requires target-system benchmark artifacts. |
| production-ha | BLOCKED | Production HA validation. | Requires a third physical failure domain. |
| kernel-e2e | UNVERIFIED | Triton or memory-pipeline end-to-end gain. | Requires correctness, profiler, microbenchmark and Agent trace evidence. |
