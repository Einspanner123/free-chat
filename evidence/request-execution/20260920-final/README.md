# Request execution confirmation: final local evidence

LOCAL_CONTRACT_ONLY. **475 passed, 12 GPU-dependency tests skipped**; 40 new cases
relative to the preceding 435-test group-reconciliation run. Ruff passed and strict mypy
passed for 112 source files. Python 3.12.14 on macOS. The lockfile hash is unchanged;
uv lock consistency and Buf breaking were not rerun because their tools were unavailable.

Measured line/branch coverage: execution contract and request ledger 100%; execution
RPC/reconciliation module about 97%; selected services/contracts combined about 89%.
This is not whole-repository or real engine coverage. See coverage.json for exact
denominators, missing lines and partial branches.

The full suite includes:

- HTTP completion retaining the sole available reservation until a separate execution
  receipt is applied; another HTTP call receives admission rejection in the meantime;
- all execution statuses against all quiescent/admission-closed flag combinations;
- tenant/request/decision/Worker/generation/engine binding, timestamp and sequence fences;
- identical receipt replay, conflicting or regressed observations, terminal-state protection;
- cancellation surviving late HTTP completion, expiry-driven abort, reconstructed ledger
  recovery, unavailable or unbound runtime retaining capacity;
- receipt/state/release-event atomicity before CAS failure and after commit-ack loss;
- full outbox rejecting release without storing a partial receipt;
- actual loopback RequestExecutionService RPCs and default Scheduler serve()/maintenance,
  including explicit local config, token rejection and stale incarnation rejection;
- Hypothesis lifecycle sequences with actual receipt-driven capacity return.

The runtime supplies deterministic **CPU fixture observations**. There is no inference
engine adapter here, no actual late-admission tombstone or physical abort execution,
no SSH/GPU/remote deployment, and no real etcd/NATS process-crash test. Default missing
execution config now intentionally retains completed reservations; see the compatibility
and deployment caveats in docs/request-execution.md before running existing demos.

tests.xml and coverage.json are actual full-regression outputs. manifest.json records
source, generated protocol, tests, docs, dependency identities and artifact SHA-256.
Git identity is null because this local mirror has no Git metadata. The intermediate
failure in sibling 20260920/ is retained and was fixed, not waived.

Reproduce in the locked CPU development environment:

```sh
ruff check .
python -m mypy libs services worker benchmarks
python -m pytest -q --tb=short --cov=freechat_scheduler --cov=freechat_gateway --cov=freechat_contracts.execution --cov=freechat_trace_replay.bus --cov-branch --cov-report=term --cov-report=json:/tmp/freechat-execution-check/coverage.json --junitxml=/tmp/freechat-execution-check/tests.xml
```

On this host the run used explicit, colon-separated source paths in PYTHONPATH for
libs/{contracts,control-store,protocol,harness-adapters}/src,
services/{gateway,scheduler,trace-replay,kv-controller,topology-agent}/src and worker/src,
because editable-package discovery in the existing .venv did not work. No runtime
dependency, lockfile or package installation was changed.
