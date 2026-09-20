# FreeChat Agent-Aware Inference Infrastructure

FreeChat turns Agent and Harness workload semantics into auditable inference
decisions for heterogeneous GPU clusters. It consumes session, branch,
pause/resume, priority and prefix-reuse signals, then decides placement,
admission, preemption and KV-cache lifecycle without taking ownership of the
agent loop.

The implementation targets Python 3.12, PyTorch, Triton and a pinned vLLM
fork. Public model traffic remains compatible with OpenAI Chat Completions,
OpenAI Responses and Anthropic Messages. Control commands use gRPC; reliable
lifecycle events use NATS JetStream; metrics and traces use OpenTelemetry.

## Current verified surface

- Versioned Agent hints and lifecycle validation.
- Capability-first placement with explainable cost breakdowns.
- Lifecycle-aware KV retain/offload/evict policy.
- Multi-tenant cache-salt derivation.
- Gateway request normalization and streaming proxy primitives.
- Synthetic control-plane validation up to 64 independent GPU workers.
- A shared lifecycle schema for four Harnesses, framework-native contract tests
  for OpenAI Agents SDK and LangGraph, and replay-safe OpenCode/OpenHands event
  bridges. Real-Harness performance remains unverified.
- etcd-compatible compare-and-swap state, JetStream subject validation and Parquet trace artifacts.
- Authenticated five-page infrastructure console with digest-pinned container builds.

Performance claims are intentionally absent until the target implementation
has produced reproducible raw evidence. See `docs/claims-ledger.md`.

## Development

```bash
uv sync --all-packages --group dev
uv run pytest
uv run ruff check .
uv run mypy libs services worker
```

Development takes place only on ross. Initialize the engine with
`git submodule update --init --recursive`; do not use `--remote` for builds.
Run `uv run python -m tools.check_source_checkout` before recording source evidence.
Use `--check-remote` to verify retrieval of the pinned fork tree.
`--release` also checks the image-digest prerequisite, not GPU/runtime acceptance.
See `docs/source-governance.md` for roles and consolidation status.

The implementation contract remains in
`agent-aware-inference-infra-plan.md` until every acceptance gate
has passed and the project owner explicitly approves its deletion.
