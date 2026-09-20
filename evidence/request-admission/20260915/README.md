# Request reservation and atomic intent local evidence

Evidence level: LOCAL_CONTRACT_ONLY. No SSH, GPU execution, deployment, real etcd/NATS
service or physical crash experiment was performed. This directory does not supersede
hardware evidence or provide a throughput, RTO, HA or Agent performance claim.

2026-09-15 local checks: **396 passed, 12 GPU-dependency skips**; Ruff and strict mypy
(106 source files) passed; uv lock consistency passed. Python 3.12.14 on macOS arm64.
Selected coverage is 86% rounded, not whole-repository coverage. The request ledger has
160 executable lines and 52 branches, all covered; gRPC server module combined coverage
is 58%, with production bootstrap and external-store paths not fully exercised.

Key cases:

- Twelve controllers sharing an in-memory CAS store compete for two one-slot Workers:
  two admissions, ten rejections, no simulated budget overcommit.
- Concurrent identical reservation/release retries, conflicting payloads, tenant and
  generation fencing, actual renewal deadlines and operation-ID deduplication.
- Expired and cancelled requests retain capacity until a trusted completion assertion.
- Fault injection before CAS and after commit with acknowledgement loss for reserve,
  renew and release; reconstruction from the same in-memory store retains state and intent.
- Publication outage, failure after publishing before ledger acknowledgement, stable
  event IDs, bounded outbox/record/renewal/byte limits, unsupported schema and contention.
- Forty Hypothesis examples of up to thirty lifecycle operations check held-capacity
  bounds; maintenance expiry continues even when publishing raises an error.
- Local Gateway ASGI to actual loopback Scheduler gRPC rejects a second request while
  the first occupies the only slot, then admits after release. Worker HTTP is mocked.
- Real loopback RPC renewal, cancel, duplicate release and cross-tenant denial; mocked
  HTTP header-wait renewal, nonstream failure and interrupted-stream cancellation.
- Synthetic 3-node/12-GPU layouts: TP=1/2/4, respectively 12/6/3 resource groups.

Reproduce from the workspace root in a Python 3.12 environment synchronized with the
lockfile and CPU development dependencies. Choose a fresh output directory to preserve
this snapshot:

```sh
ruff check .
mypy libs services worker benchmarks
uv lock --check
pytest -q --cov=freechat_scheduler --cov=freechat_gateway --cov=freechat_trace_replay.bus --cov=freechat_contracts.cache_identity --cov-branch --cov-report=term --cov-report=json:/tmp/freechat-request-check/coverage.json --junitxml=/tmp/freechat-request-check/tests.xml
python -m freechat_scheduler.local_cluster --output /tmp/freechat-request-check/layouts.json
```

`tests.xml`, `coverage.json` and `layouts.json` are outputs, not invented result tables.
`manifest.json` pins source, tests, protocol, lockfiles and artifact SHA-256 hashes. This
local mirror has no Git metadata; the manifest records null rather than an assumed SHA.
Buf breaking, real engine completion/abort, tokenizer/budget producers, reconciler,
authenticated RPC ownership, persistent consumer deduplication and ledger compaction
remain open. See `docs/review-hardening.md` and the retained implementation plan.
