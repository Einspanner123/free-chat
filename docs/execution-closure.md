# Execution closure: ownership, integration boundary and local group reconciliation

## Scope and source audit

This is the first resource-group slice of the execution-closure plan, not completion
of the entire Worker stage. Work is local only: no SSH, H100, real model launch or
etcd/NATS cluster. The inspected mirror is not a complete upstream checkout. The
following is a local integration audit, not a claim that upstream lacks a feature.
The dependency and fork identities remain those in `versions.lock.yaml`.

| Capability | Inspected local source | Decision |
|---|---|---|
| Inference, attention, block allocation | fork cache-manager/hooks and Worker bridge | Keep engine ownership; do not implement another engine |
| Cache lifecycle events | `worker/src/freechat_worker/vllm_adapter.py` | Bridge currently maps allocate/free/hit/evict; not a terminal request receipt |
| Engine metrics | `worker/src/freechat_worker/telemetry.py` | Aggregate usage/transfer counters are not per-rank admission budgets or a residency catalog |
| Request reservations | `services/scheduler/src/freechat_scheduler/request_ledger.py` | Keep request ownership separate from cached KV and transfer reservations |
| Static resource-group lifecycle | `services/scheduler/src/freechat_scheduler/resource_groups.py` | Reuse existing CAS ledger and states; add reconciliation, not a second allocator |
| Worker/engine lifecycle | new `group_runtime.py`, `group_reconciler.py` | Implement typed command/receipt contract and local integration; actual engine driver still required |
| Mooncake/AgentX and fork patch parity | No complete local upstream checkout audit in this slice | Remain follow-up gates; no dependency replacement or performance claim |

## Implemented path

An explicitly configured Scheduler loads `GroupRuntimeConfig` through
`FREECHAT_GROUP_RUNTIME_CONFIG`, constructs its group view, and runs a periodic
reconciler alongside request-ledger maintenance. No configuration means grouped
Workers still fail closed. Existing group reservations are required; the config
does not create them, renew ownership or synthesize hardware evidence.

The configuration contains `mode: local-contract`, `owner_id`, a validated `inventory`,
optional measured `links`, and `endpoints` keyed by node ID. Each endpoint has an
`address` and secret `token` (at least 32 characters). Keep such files out of Git.
Addresses are restricted to IPv4 loopback. No default token, remote insecure channel,
or fallback fixture driver is installed. The canonical working configuration example
is constructed in `test_default_serve_uses_configured_group_view`, with synthetic
inventory and an ephemeral test-only token. The topology-intent file is not that config.

`GroupRuntimeService.Apply` carries strict Pydantic command and receipt payloads in
protobuf envelopes. This is a first local contract: Buf does not check the inner JSON
schema, so its compatibility needs model-level tests and future schema/version work.

- Commands bind immutable spec, node, owner, group generation, expected engine instance
  and action. Their deterministic operation ID survives retry/reconstruction.
- START/INSPECT can establish READY only with matching model/revision/TP/GPU set,
  engine identity, fresh healthy telemetry and a complete, unique per-rank budget set.
  The admission budget must equal the minimum rank budget and cannot exceed declared
  physical rank capacity. Zero is valid telemetry but grants no positive-size request.
- DRAIN does not advance until the runtime reports drained and quiescent.
- STOP does not release allocation until STOPPED plus quiescent is reported. Unknown,
  unavailable, stale and mismatched replies retain ownership. Lease expiry triggers
  quarantine/stop, not immediate reassignment.
- Each active owned group is reconciled concurrently with a two-second command deadline.
  A failed group is hidden from the route view; others can become available. The view
  is invalidated while refreshing and timestamped before the scan, so a slow scan does
  not make old observations appear fresh. Existing Scheduler age checks remain active.
- Each local Runtime RPC checks a loopback peer and configured bearer token. Node/owner
  scope must match. This is a CPU-test identity boundary, **not SPIFFE/mTLS**.
- Before side effects, the runtime records incarnation/spec, GPU ownership and monotonic
  command phase in its provided CAS store. Delayed START cannot reopen a stopped
  incarnation, an old STOP cannot target a newer generation, and overlapping groups
  cannot start while prior ownership is uncertain. Driver observations cannot silently
  change an already recorded engine instance.

## Ownership and safety limits

The runtime store requires one active node supervisor for that namespace. This is not
distributed node-agent leader fencing. A real driver must durably identify processes
by incarnation, validate local GPU ownership, make side effects idempotent, and inspect
actual process/collective termination before claiming quiescence. A supplied receipt is
not proof that a physical engine stopped. Tests use a deliberately non-serving CPU
fixture; no concrete vLLM process/container driver is installed.

Control-plane group leases are not renewed automatically by successful polling: the
owner must renew them. Lost runtime state fails closed on unknown inspect/stop; explicit
reconciliation is needed rather than guessing an engine is gone. Runtime fence history
has no compaction yet. Group state, registry state and lifecycle events still do not have
one atomic publication boundary. The periodic view is not a linearizable cross-service
transaction, and inflight inference still requires engine-side admission/fencing.

The request-execution follow-up changes `Release` into a transport-completion intent;
capacity now requires an incarnation-bound terminal/quiescent/admission-closed receipt.
See `docs/request-execution.md` for the default service integration and local evidence.
Group stop receipts **do not release request reservations**. Real engine termination,
admission fencing and reservation-aware rank-budget accounting remain incomplete.

## Acceptance and remaining order

Local tests exercise the actual default `serve()` bootstrap, a loopback Runtime gRPC
server, Scheduler routing and request reservation; no monkeypatched routing algorithm.
Injected stores/drivers model acknowledgment loss, unavailable engines, invalid rank
budgets, owner/generation/spec/instance fencing, drain/stop waiting and supervisor object
reconstruction. This does not constitute OS process crash or physical termination tests.
Final source-pinned evidence: `evidence/group-reconciliation/20260920-final/`.

1. The local per-request binding and QUERY/ABORT receipt contract is now implemented;
   complete its actual engine admission fence, execution observation and cancellation driver.
2. Engine-backed tokenizer/layout and rank-budget producer, request/cache/transfer
   reconciliation, concrete node driver and internal mTLS identities.
3. Worker/topology/outbox atomicity, multi-replica registry views, compaction and real
   local etcd/NATS service interruption tests.
4. Actual cache residency directory; Session-Sticky baseline and work-based cost model.
5. AgentX replay and all four Harness lifecycle/forecast inputs; real read models/UI.
6. Separately authorized physical H100 topology/model/profiler/performance acceptance.

The main plan remains active. No GPU/HA/RTO/performance claims are added.
