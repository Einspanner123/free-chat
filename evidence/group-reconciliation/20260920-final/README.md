# Local resource-group reconciliation evidence

Evidence level: LOCAL_CONTRACT_ONLY. **435 tests passed; 12 GPU-dependency tests
skipped.** Ruff passed; strict mypy passed for 109 source files. The dependency lock
was not changed (SHA-256 matches the 2026-09-15 manifest). `uv lock --check` and Buf
breaking were not rerun: those executables were unavailable. Python 3.12.14 on macOS.

This snapshot adds 39 tests/parameter cases relative to the previous 396-test local
baseline. It validates:

- Group readiness, routing, draining, stop confirmation and allocation reuse;
- command ack loss and object reconstruction without duplicate fixture instances;
- owner/node/generation/operation/engine fences, complete per-rank budgets and freshness;
- unknown and unavailable runtime observations retaining allocation and hiding route views;
- persisted local runtime phases rejecting delayed starts/stops and overlapping groups;
- slow group timeout without preventing independent group reconciliation;
- actual Scheduler `serve()` startup with explicit runtime config and loopback gRPC;
- local RPC token denial, scoped owner identity, malformed commands and nonlocal peer denial.

The two new modules have combined line/branch coverage rounded to 98% (reconciler) and
99% (runtime). Full selected measurement scope is about 89%, not full-repository coverage.
Read `coverage.json` for exact denominators and missing paths. All three synthetic
3x4-GPU layouts also passed the existing loopback layout exercise.

Important limits: the runtime driver is a non-serving CPU fixture. No model/container
is launched, no OS process termination is proven, and per-request completion/abort
receipts remain unimplemented. The token boundary is loopback-only, not production
SPIFFE/mTLS. State stores are in memory; reconstruction tests are not real etcd or
node crash evidence. See `docs/execution-closure.md` for the complete remaining gates.

Artifacts:

- `tests.xml`: full regression results, including explicit skip reasons;
- `coverage.json`: selected source line/branch measurements;
- `layouts.json`: synthetic TP=1/2/4 layout exercise, not measured H100 behavior;
- `manifest.json`: source/protocol/tests/lock/docs/artifact SHA-256 and check statuses.

The local mirror has no Git metadata; no commit was invented. Earlier draft output
is preserved in sibling `20260920/` and is not used for final acceptance.

Reproduction uses the locked CPU development environment, `ruff check .`,
`python -m mypy libs services worker benchmarks`, and:

```sh
python -m pytest -q --cov=freechat_scheduler --cov=freechat_gateway --cov=freechat_trace_replay.bus --cov-branch --cov-report=term --cov-report=json:/tmp/freechat-group-check/coverage.json --junitxml=/tmp/freechat-group-check/tests.xml
python -m freechat_scheduler.local_cluster --output /tmp/freechat-group-check/layouts.json
```

This host's editable package path discovery did not work, so the actual run supplied
PYTHONPATH explicitly with `libs/{contracts,control-store,protocol,harness-adapters}/src`,
`services/{gateway,scheduler,trace-replay,kv-controller,topology-agent}/src`, and `worker/src`
(each path expanded, colon-separated). No environment package or lockfile was modified.
