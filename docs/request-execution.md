# Request execution confirmation and reconciliation

## Implemented boundary

Gateway transport completion is no longer permission to release execution capacity.
`Release` records `completion_pending`; `Cancel` records `cancel_requested`. Both hold
the full reservation. A late completion cannot downgrade an expired/cancelled request.
Completion-pending requests also expire, allowing the reconciler to request an abort.

The immutable route now records `engine_instance_id` from the selected Worker telemetry.
Optional protobuf RouteDecision field 14 and Gateway parsing carry it end to end; the
Gateway forwards it as `x-freechat-internal-engine-instance-id`. This header must be
enforced by a future real engine adapter; header transport alone does not fence execution.
An unbound historical route cannot be released through the execution receipt path.

`RequestExecutionService.Observe` transports strict command/receipt models using the
same local-only transport boundary as group-runtime validation. Commands bind tenant,
request, decision, Worker generation, engine instance and QUERY/ABORT action. Stable
operation IDs identify repeated queries or aborts for the same incarnation.

| Reservation state | Background action | Capacity |
|---|---|---|
| active | QUERY | Held |
| completion_pending | QUERY, then ABORT after expiry | Held |
| cancel_requested | ABORT | Held |
| expired | ABORT | Held |
| released | No further polling | Returned exactly once in the ledger |

An execution observation releases capacity only if all three conditions hold:

1. State is COMPLETED, ABORTED or NOT_ACCEPTED;
2. `quiescent` asserts no execution remains for that request incarnation;
3. `admission_closed` asserts durable rejection of late data-plane arrivals/retries.

UNKNOWN/RUNNING never release capacity. A negative lookup, HTTP EOF, an abort signal
being sent, or a group being stopped is not sufficient proof. In particular, a request
that has not arrived yet can appear absent, so NOT_ACCEPTED without an admission fence
must retain capacity.

## Atomicity, ordering and provenance

The request ledger checks tenant/request/decision/Worker/generation/engine identity,
timezone-aware observation freshness, and monotonic observation sequence. Exact
duplicate receipts are no-ops; a reused sequence with changed contents or a lower
sequence fails closed. Released requests cannot be reopened by later observations.
The driver must persist observation numbering across restarts; resetting it is not a
reason to relax the consumer fence. Receipt freshness is checked for new observations;
replayed identical receipts can remain idempotent without modifying terminal state.

The last receipt, release state and `lease.released` event intent share a single CAS.
Audit events include the receipt. Repeated observations of the same nonterminal status
advance the stored sequence without emitting another status event on every poll.
Before-commit failure changes neither state nor event; lost commit acknowledgement
retains both, and an identical retry does not create another release event.

## Default service integration

`FREECHAT_REQUEST_EXECUTION_CONFIG` points to an operator-provided JSON config containing
`mode: local-contract` and an `endpoints` map keyed by worker ID. Each endpoint contains
an IPv4 loopback `address` and a secret `token` of at least 32 characters. Do not commit
real credentials. The runnable fixture configuration is built in
`test_default_scheduler_maintenance_releases_only_after_worker_proof`.

The default Scheduler maintenance loop expires requests and then reconciles execution
observations. Polling is limited to eight concurrent operations with two-second
timeouts. RPC errors, missing endpoints, unbound engine identities, malformed or
mismatched observations keep reservations held. No configuration means there is no
automatic release fallback: completed calls remain pending and can exhaust admission
capacity until a configured execution adapter supplies valid observations.

The generated RPC method name `Release` is retained, but its semantics changed. The
Gateway now accepts completion_pending/cancel_requested/expired/released responses as
recorded intent states; older clients that demand immediate released must be updated.
Do not mix Scheduler binaries with different release semantics against one ledger.
Previously released records retain their historical meaning and are not retroactively
converted into Worker execution proofs. The ledger is not cleared or silently migrated.

## Evidence and remaining gates

Source-pinned final local evidence: `evidence/request-execution/20260920-final/`.
Tests include HTTP completion retaining a full one-slot budget, real loopback RPCs and
default service maintenance, terminal-state/flag combinations, identity fences, stale
or conflicting receipts, cancellation/expiry, reconstructed ledger objects, outbox
exhaustion and CAS acknowledgement loss. CPU fixture observations do not constitute
actual model execution or physical process termination.

Still required:

- A real engine admission/execution backend enforcing the exact decision and engine
  identities on incoming inference. The local Worker journal now supplies durable
  tombstones and observation sequencing when all ingress passes through it; see
  `docs/worker-admission.md` and `evidence/worker-admission/20260920/`. It is not wired
  into the vLLM frontend or enabled by the default Worker startup;
- cancellation delivery and physical quiescence verification in that engine;
- reservation-aware per-rank budget reporting and request/cache/transfer reconciliation;
- tokenizer/template-exact token counts, internal mTLS and authenticated Worker ownership;
- actual etcd/NATS process restart tests, ledger compaction and production rollout checks.

The local bearer token authenticates the RPC caller in a controlled test environment;
it is not server-authenticated TLS or a physical execution attestation. Group stop does
not bulk-release individual calls. All budget accounting remains conservative and may
double-count engine occupancy until reservation-aware telemetry is implemented. No
H100, GPU throughput, HA, RTO or exactly-once inference claim is added.
