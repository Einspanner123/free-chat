# Local H100-target control-logic gate

Date: 2026-09-14. Evidence level: **LOCAL_SIMULATION_ONLY**.

This is a local source snapshot, not a deployment or hardware acceptance. No SSH,
H100 execution, model inference, remote cluster changes or cloud-model call was made.

- Python 3.12.14, macOS ARM64, checked-in uv lock; GPU extras absent.
- Full Python test collection: 292 passed, 12 GPU-dependency tests skipped, zero failures.
- Ruff passed; strict mypy passed for 100 source files; `uv lock --check` passed.
- Resource groups, parallel admission and forecast validation: 100% measured line and
  branch coverage. Scheduler package combined line/branch coverage: 81% (rounded),
  including the unexecuted external live-validation module. This is not whole-repository
  or WebUI coverage.
- Loopback RPC simulation: TP=1/2/4 -> 12/6/3 groups -> 24/12/6 routed/released requests.
  Synthetic inventory and model/engine identities; no inference endpoint was contacted.
- Coverage does not prove all distributed interleavings, authentic engine shutdown,
  etcd/NATS/MinIO failure handling, GPU correctness, HA, or task performance.
- Buf breaking comparison and WebUI/browser tests were not run in this gate.

`tests.xml` and `coverage.json` contain the final test results. `layouts.json` is a separate
CLI run. `manifest.json` records source/lock/artifact SHA-256 values and exact coverage.
The source tree is a local mirror without Git metadata; there is no new commit identity.

During validation, a stale generated CostBreakdown schema was repaired. A subsequent
loopback test exposed lost forecast metadata and active resume horizons across gRPC;
the additive protocol field and both transport endpoints were fixed and tested.
An unrelated local environment issue repeatedly marked editable `.pth` files hidden;
the final gate used a fresh temporary virtual environment from the same lock.

Remaining work is recorded in `docs/h100-local-implementation.md`. Keep the main plan.
