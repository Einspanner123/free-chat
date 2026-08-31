# Inference engine boundary

The project maintains an independent vLLM fork at the exact coordinates in
`versions.lock.yaml`. The public upstream remote remains read-only reference;
the fork branch is pushed to a separate bare repository under the FreeChat
artifact root so its commit is recoverable without relying on an uncommitted
working tree.

## Fork changes

The current fork slice adds only internal request metadata and an opt-in KV
cache hook. It does not alter vLLM's HTTP protocols, PagedAttention,
continuous batching, attention backends, model implementations or existing KV
event/offload connectors.

Lifecycle metadata contains authenticated tenant/cache identity, task, agent,
branch and call identity, lifecycle state, resume prediction, policy priority,
and worker/cache generations. Unknown fields and invalid lifecycle combinations
fail closed. The hook emits completed allocation, prefix-hit, free and explicit
eviction observations. With no hook configured, the manager uses a no-op path
and skips event construction.

The FreeChat worker bridge:

- validates worker and cache generation before accepting an event;
- drops uncorrelated events rather than guessing a tenant;
- maps block groups to explicit token and byte layouts;
- writes to a bounded, thread-safe, non-blocking queue;
- exposes accepted, stale, uncorrelated, malformed and overflow counts.

## Evidence and remaining gate

- Upstream baseline: `9c22668436a4d94aab87ea74a220e060415cf1d8`.
- Fork commit: `1c191f8632f1bfdd57d5e1de3b749b60f824f98a`.
- Fork tests: 90 prefix-cache tests plus four lifecycle/request tests passed on
  ross. The focused hook run adds one manager lifecycle integration test.
- Main repository: 56 tests passed in the locked CUDA 13 environment, including
  cross-thread buffer and stale/uncorrelated bridge behavior.

The hook factory has not yet been exercised inside a digest-pinned worker image
serving a real model. Until that model-server test, cancellation/restart trace,
and event-to-controller flow pass, the engine integration remains incomplete
and cannot be presented as a serving-system capability.
