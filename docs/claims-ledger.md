# Claims ledger

Only entries marked `VERIFIED` may appear as performance or production claims
in project-facing material.

| Claim ID | Status | Claim | Evidence |
|---|---|---|---|
| architecture-contract | VERIFIED | The repository defines versioned Agent hints, capability-first placement and lifecycle-aware KV policy. | Unit and scale tests in this repository. |
| harness-contracts | VERIFIED | Four named Harness adapters emit the same lifecycle and identity depth. | `libs/harness-adapters/tests/test_adapters.py`. |
| control-images | VERIFIED | Gateway and WebUI images build from locked base-image digests and start as non-root services. | workstation container build and in-container health checks, 2026-08-31. |
| control-state | VERIFIED | etcd compare-and-swap failure, JetStream subject injection and Parquet artifact hashing have executable tests. | `libs/control-store/tests` and `services/trace-replay/tests`. |
| gpu-performance | UNVERIFIED | GPU or Agent-task performance improvement. | Requires target-system benchmark artifacts. |
| production-ha | BLOCKED | Production HA validation. | Requires a third physical failure domain. |
| kernel-e2e | UNVERIFIED | Triton or memory-pipeline end-to-end gain. | Requires correctness, profiler, microbenchmark and Agent trace evidence. |
