# Local correctness follow-up to the interviewer review

Scope: local implementation and CPU/loopback validation only. No SSH, deployment,
GPU execution, new performance claim, or main-plan deletion is authorized here.

## Implemented contracts

### Per-candidate KV feasibility and locality

ModelCapability now declares `kv_admission_bytes_per_token_per_rank` (the largest
rank allocation per token, including replicated KV) and `kv_block_size_tokens`.
WorkerTelemetry declares `kv_admission_available_bytes_per_rank`: the minimum
usable KV budget across all ranks, after excluding live/non-reclaimable blocks
and engine reservations. It is neither aggregate HBM nor CUDA free bytes.

For each candidate, the Scheduler computes:

`ceil((input_tokens + output_tokens) / block_size) * block_size * bytes_per_token_per_rank`

This ignores unproven prefix hits and never divides KV size by TP. Unknown geometry
or budget rejects the candidate. Reported zero budget is known, not missing.
Client `estimated_kv_bytes` cannot lower this feasibility requirement. Offload transfer
size remains a separate input-prefix estimate; it is not the output-inclusive budget.
Eviction externality uses the same derived per-rank demand instead of a default zero.

Gateway takes the routing origin only from `FREECHAT_ORIGIN_NODE_ID`; optional
RouteRequest field 8 carries it over gRPC. It identifies the administrator-defined
origin for locality policy, not the physical location of an untrusted HTTP client.
Unknown locality rejects requests or Workers forbidding remote execution. Permissive
requests can still route, but missing locality invalidates cost estimates explicitly.
Scheduler admission/unavailability RPC errors map to HTTP 503, invalid input to 422,
deadline expiry to 504, and other RPC failures to 502; no Worker request is sent on rejection.

**Important limits:** Gateway token counting is still heuristic; this gate is conservative
only relative to supplied token counts and Worker geometry/budget. Exact tokenizer/template
accounting and engine-side final admission remain required. The request ledger below
now subtracts outstanding reservations atomically against reported snapshot budgets;
this still is not an engine-memory or multi-replica registry-consistency guarantee.
The current generic Worker metrics sampler does not provide the new admission budget.
Do not invent that budget from free HBM or unscoped aggregate cache counters: a Worker
cache/rank reconciler must report it. Registration alone now intentionally cannot admit
requests before suitable telemetry arrives.

### Cache namespace semantics

All scopes remain bounded by authenticated tenant and privacy_domain. The canonical
identity is serialized as JSON before HMAC, avoiding separator ambiguity:

| Scope | Additional identity |
|---|---|
| GLOBAL | None: sharing is tenant-local, never cross-tenant |
| HARNESS | Harness |
| TASK | Harness, task, session |
| AGENT | Harness, task, session, agent |
| BRANCH | Harness, task, session, agent, branch |
| PRIVATE | Harness, task, session, agent, branch, logical call |

Retries with the same logical call retain its PRIVATE namespace; another call does
not. Different scopes have different namespaces. Parent-prefix sharing across child
branches is not inferred or forced; segmented lineage/prefix sharing remains future work.
The namespace calculation changes existing cache salts, so cache warmth is not preserved.
These tests establish the Gateway namespace contract, not a timing-side-channel proof.

### Registration and heartbeat fencing

Both registries reject decreasing generation and conflicting same-generation capabilities.
Equivalent same-generation registration retries retain existing telemetry and engine
identity. A higher generation creates a new incarnation. Persistent registration and
heartbeat validate the authoritative store record and update it with bounded CAS retries;
a stale controller's heartbeat cannot overwrite a newer incarnation. Failed writes do
not publish a speculative new worker record in that controller's local cache.

**Limits:** generation monotonicity is not authorization. Authenticated ownership/issuance,
cross-replica watch/reconciliation, and atomic worker/topology-generation/event publication
remain unfinished. The topology counter and event outbox are still separate writes.
Do not label this change multi-replica HA or exactly-once lifecycle processing.

## Remaining review findings, in implementation order

2026-09-20 follow-up: a configured loopback group reconciler and typed node receipts now
reach default service startup; see `docs/execution-closure.md`. The list below remains
open for actual Worker deployment/measurement, per-request completion and production identity.

Local gate: 370 Python tests passed, 12 GPU-dependency tests skipped; Ruff, strict mypy
(104 source files) and lock consistency passed. KV feasibility and cache identity have
100% measured line/branch coverage; this does not establish hardware or deployment behavior.

The counts above describe the 2026-09-14 snapshot. The following request-admission
implementation has a separate source and evidence snapshot.

1. Resource-group deployment reconciler, trusted Worker/rank budget publisher, startup,
   readiness/drain/stop acknowledgements and periodic group/worker views.
2. Worker-confirmed request completion/abort, reservation-aware telemetry, exact tokenizer
   accounting, authenticated ownership and multi-replica registry watch/reconciliation.
3. Atomic Worker/topology/event publication, persistent consumer deduplication and bounded
   request-ledger compaction/sharding; real etcd/NATS interruption acceptance remains open.
4. Per-call forecast producer injection and explicit lifecycle notification for four Harnesses.
5. Actual console read models, managed deployment bootstrap and cloud-provider boundary.

The implementation plan retains all these gates. Historical evidence under
`evidence/h100-local/20260914/` belongs to that source snapshot; follow-up results are
stored separately under `evidence/review-hardening/20260914/`.

## Request admission and atomic lifecycle intent

The default Scheduler gRPC path now uses `RequestLedger`, a bounded CAS record at
`/freechat/request-ledger`. Tenant-scoped idempotency identity, request fingerprint,
decision, per-rank KV reservation, expiry and pending event intent are committed together.
CAS losers re-read outstanding reservations and re-plan. Admission subtracts held bytes
and adds held request counts before capability filtering and least-load placement.

An active identical retry returns the same decision; changed payloads, expired requests
and terminal retries are rejected. This deduplicates reservations, **not backend execution**.
Renewal has an actual deadline and operation-ID deduplication. Release is one-time and
requires matching tenant, Worker and generation. Public protocol RouteDecision field 13
reports `reserved_kv_bytes_per_rank`.

| State | Holds capacity | Allowed next action |
|---|---|---|
| active | Yes | Renew, cancel, expire or trusted completion release |
| completion_pending | Yes | Query execution; abort after expiry, or receipt-confirmed release |
| expired | Yes | Cancel or trusted completion release; no renewal |
| cancel_requested | Yes | Trusted completion release; no renewal |
| released | No | Idempotent release/cancel only; no re-admission under the same key |

Gateway renews during nonstream requests and streaming header/prefill waits as well as
stream bodies. Completed responses now record completion_pending, not release. Timeouts, transport failures and
interrupted streams assert cancel; cancel does not claim the engine has stopped.
The local QUERY/ABORT reconciliation and receipt gate are documented in `docs/request-execution.md`;
the local Worker-side durable admission journal is described in `docs/worker-admission.md`,
but actual engine ingress/completion integration is still missing. Internal
tenant fields are checked for consistency but are not a substitute for authenticated RPCs.

Maintenance publishes persisted event intents then acknowledges them in the ledger;
publication failure does not prevent expiry processing. Publish-ack loss can repeat an
event with the same ID and sequence. Delivery is at least once, not exactly once; consumers
must handle duplicate/out-of-order events. With no publisher configured, intents remain
pending and bounded capacity eventually rejects writes rather than silently dropping them.
The atomic boundary applies to request state, not Worker registration or topology writes.

Bounds: 10,000 retained records, 20,000 pending events, 4,096 renewal operation IDs per
request and 1 MiB serialized ledger by default; the first reached bound fails closed.
There is no tombstone compaction or sharding yet. The single-key design is a local
correctness foundation, not sustained 64-GPU control-plane throughput evidence. Held
reservations count across Worker generations conservatively. Telemetry may already account
for executing requests, so double counting is possible until reservation identities are
reconciled. A stale registry snapshot is not fixed by request CAS alone.

Existing decision-only `/freechat/leases/` records cause explicit reconciliation failure
when the request ledger is absent; they are not silently erased or treated as safe budgets.
The historical `live_validation.py` probe is not acceptance for this schema: its synthetic
registration lacks rank-budget telemetry and its fallback reads decision-only records.
It must be revised with a trusted budget producer before external validation is resumed.

Local evidence is isolated in `evidence/request-admission/20260915/`: concurrent admission,
duplicate calls, expiry boundaries, random lifecycle sequences, failures before CAS and
after commit/ack loss, outbox replay, tenant/generation fences, Gateway disconnect semantics,
and real loopback gRPC. No physical GPU, etcd/NATS cluster, deployment, HA or performance
acceptance is implied. The next implementation is the reconciler and acknowledgement path
listed above; the main plan remains active.
