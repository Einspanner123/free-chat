# Three-node H100 resource groups: local implementation and verification

## Scope and authority

The owner supplied 3 nodes x 4 H100, with NVLink available. Treat NVLink as intra-node
only until measured. Per-GPU memory, NVSwitch/PCIe connectivity, network and failure
domains remain unknown. `deploy/h100/topology-intent.json` records intent, not deployable
hardware evidence. No SSH, GPU execution or cluster changes are authorized in this phase.

A remains conditional admission-time native KV offload. This change does not implement
B, change the dependency/GPU lock, replace vLLM, or claim new inference performance.

## Implemented control logic

- `Inventory`: globally unique GPU identities, owning node, capacity (unknown allowed,
  but unknown capacity cannot pass admission), health and compute capability.
- `GroupSpec`: immutable desired model/revision, GPU set, TP=1/2/4, per-GPU memory budget
  and operator-supplied, expiring parallel evidence. Cross-node groups are rejected.
- `GroupController`: entire allocation in one revision-fenced CAS ledger via the existing
  `KeyValueStore`. Concurrent controllers cannot reserve overlapping partial groups.
  The ledger is authoritative; no successful mutation is published before its CAS.
- Monotonic allocation generation, owner fence, previous-generation check on group reuse,
  idempotent reservation retries and explicit renewal. Delayed creation/release messages
  cannot revive or release another incarnation of a group.
- Desired configuration and observed lifecycle are separate. RESERVED -> STARTING ->
  READY -> DRAINING -> STOPPING -> RELEASED; failures retain ownership until stopped.
  Expiry enters QUARANTINED rather than immediately reassigning potentially live GPUs.
  READY requires an engine/readiness acknowledgement; RELEASED requires quiescence acknowledgement.
- Worker capability JSON binds resource-group ID/generation and GPU set. Scheduler reads
  one group view per decision and checks freshness, READY state, expiry, worker/engine
  identity, model revision, TP configuration, node and all GPU members before routing.
  Multi-GPU workers without group binding fail closed; existing single-GPU clients remain compatible.
- ParallelPlanner now rejects duplicate/empty GPU identities, nonfinite cost measurements,
  duplicate/self edges and disconnected topology, rather than trusting an edge count.

Evidence references/acknowledgements are internal caller assertions, not cryptographic
attestation. In the original 2026-09-14 snapshot, the default gRPC launcher had no group-view reconciler:
multi-GPU registration alone will not enable routing. The local validation runner wires
the controller and snapshot explicitly. The 2026-09-20 follow-up adds explicit loopback
runtime configuration to the default launcher; see `docs/execution-closure.md`. This
is still not actual Worker deployment, engine termination or physical budget verification.

## Reproduce locally

Use Python 3.12 and the checked-in uv lock, without the GPU extras:

```sh
uv sync --locked --all-packages --group dev
uv run --no-sync pytest -q --cov=freechat_scheduler --cov-branch --cov-report=term-missing
uv run --no-sync ruff check .
uv run --no-sync mypy libs services worker benchmarks
uv run --no-sync python -m freechat_scheduler.local_cluster --output /tmp/freechat-local-layouts.json
```

On the local macOS checkout, editable `.pth` files repeatedly acquired the filesystem
hidden flag; Python then skipped their import paths. The final run uses a separate
temporary environment with the same lock, not a PYTHONPATH workaround or dependency
upgrade. If this environment issue recurs, before the commands above use:

```sh
export UV_PROJECT_ENVIRONMENT="$(mktemp -d /tmp/freechat-local-venv.XXXXXX)"
```

The runner refuses to overwrite an existing result. It starts only loopback gRPC,
simulates registration/heartbeat for 12, 6 and 3 resource groups, and checks routing
and release of 24, 12 and 6 requests respectively. Synthetic 80 GiB capacity and positive
parallel results are marked SIMULATED; they are not measured H100 specifications.
No model is loaded and the returned fake inference endpoint is never contacted.

The protobuf Python outputs were regenerated to fix a local mismatch in CostBreakdown
fields discovered by the loopback test. Groups use the existing capability JSON carrier.
Separately, AgentHints gains additive metadata field 26: forecast identity, expiry and
evidence reference previously disappeared during gRPC transport. Gateway now serializes
these fields and Scheduler restores them, including an active call's resume horizon.
Loopback tests check rejection of an expired forecast and round-trip preservation of
metadata, a positive horizon, explicit zero and absence. Provenance is not authentication.

## Test coverage scope

Final local gate (2026-09-14): **292 passed, 12 skipped**, Ruff passed, strict mypy passed
for 100 source files, and uv lock consistency passed. Resource-group allocation/lifecycle,
parallel admission and forecast validation each have 100% measured line and branch
coverage; the full Scheduler package combined coverage is 81% rounded. The other services
run in the regression suite but these percentages do not measure their coverage.
Raw results and source hashes: `evidence/h100-local/20260914/`. Buf breaking comparison
and WebUI/browser validation have not been performed in this gate.

Local tests include all three layouts, schema boundaries, unknown/missing/unhealthy
GPU rejection, insufficient memory, cross-node denial, stale/missing parallel evidence,
disconnected topology, competing-controller CAS retries, idempotency, snapshot isolation,
store failure, bounded contention, controller restart, renewal, expiry/quarantine,
reallocation fences, all 64 lifecycle state pairs, and randomized overlapping allocations.

Routing tests cover missing/stale/naive/future views; unready/expired groups; mismatched
group/worker generation, worker, node, GPU membership, engine, model revision and TP/PP;
and real loopback Register/Heartbeat/Route/Release under TP=1/2/4 simulations.
An allocation's engine identity cannot be replaced by a conflicting transition, including
an otherwise idempotent READY retry. The CLI tests check evidence labeling and refusal
to overwrite an existing result.

Coverage percentages describe executed Python lines/branches only. They do not cover
all distributed interleavings, malicious acknowledgements, real process termination,
hardware failures, or physical GPU behavior. Synthetic simulation is not a throughput
benchmark and does not populate resume performance claims.

## Remaining implementation and hardware gates

1. Authenticated deployment reconciler for actual worker launch, readiness, draining,
   stop acknowledgement, engine fences and periodic group-view publication; Helm/operator
   wiring and actual topology discovery. No live process launch is implemented here.
2. Durable lifecycle events/outbox, compaction and capacity limits for the group ledger,
   inventory-change reconciliation and replacement-node admission. A changed inventory
   currently refuses access rather than silently forgetting allocations.
3. Actual etcd transaction/watch/partition/failover tests. Local concurrent tests use
   an in-memory CAS adapter and must not be presented as three-node etcd HA evidence.
4. A-path prediction, native block-aligned transfer plans and reconciled KV residency;
   real Gateway/Scheduler/Worker plus four-Harness, three-model acceptance.
5. H100 P2P/NCCL/interconnect/memory measurements, TP gain and communication baselines,
   process/node/network faults and physical failure-domain qualification.
6. WebUI resource-group lifecycle views and browser tests. This change does not establish
   complete WebUI coverage or change the existing inference/playground UI.

Do not turn this local test gate into a hardware VERIFIED claim. Keep the main plan;
deletion still requires complete acceptance and explicit owner authorization.
