# Claims ledger

Only entries marked `VERIFIED` may appear as performance or production claims
in project-facing material.

| Claim ID | Status | Claim | Evidence |
|---|---|---|---|
| architecture-contract | VERIFIED | The repository defines versioned Agent hints, capability-first placement and lifecycle-aware KV policy. | Unit and scale tests in this repository. |
| gpu-performance | UNVERIFIED | GPU or Agent-task performance improvement. | Requires target-system benchmark artifacts. |
| production-ha | BLOCKED | Production HA validation. | Requires a third physical failure domain. |
| kernel-e2e | UNVERIFIED | Triton or memory-pipeline end-to-end gain. | Requires correctness, profiler, microbenchmark and Agent trace evidence. |
