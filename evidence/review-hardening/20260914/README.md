# Local review-hardening gate

Evidence level: LOCAL_CONTRACT_ONLY. Date: 2026-09-14.

Full Python regression: **370 passed, 12 GPU-dependency tests skipped, zero failures**.
Ruff passed, strict mypy passed for 104 source files, and `uv lock --check` passed.
Python 3.12.14 on local macOS; no SSH, GPU execution or deployment was performed.

Coverage scope is Scheduler plus cache identity, not the whole repository. KV feasibility
and cache identity modules have 100% measured line/branch coverage; registration module
combined coverage is 89%, and the combined selected scope is 84% rounded. No inference
speed, capacity guarantee, distributed HA or timing-side-channel result is implied.

The new checks include scope/identity matrices, block alignment and output budget for
TP=1/2/4, unknown/zero/insufficient budgets, unknown/forbidden locality, monotonic and
idempotent registration, stale-replica heartbeats, competing CAS writers, store failure
and bounded retry. A local HTTP Gateway -> loopback gRPC Scheduler -> mocked inference
endpoint test verifies trusted locality, branch salts and rejection before forwarding.
A real loopback Register RPC rejects a decreasing Worker generation.

`tests.xml` and `coverage.json` are the test-run outputs. `layouts.json` separately checks
the three synthetic layouts. `manifest.json` contains source/lock/artifact SHA-256 hashes.
The workspace is a local mirror without Git metadata; no remote commit is asserted.

Critical remaining requirements are documented in `docs/review-hardening.md`: exact token
accounting, trusted Worker rank-budget reporting, request capacity reservations, atomic
state/outbox publication, authenticated generation ownership, multi-replica reconciliation,
deployment coordinator, Harness forecast injection and actual console state.
