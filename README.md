# FreeChat Agent-Aware Inference Infrastructure

FreeChat maps Agent/Harness lifecycle signals to admission, placement and KV-cache
decisions. It reuses vLLM's model execution, HTTP protocols, continuous batching
and cache machinery; it does not implement an Agent loop or a second inference engine.

The implementation uses Python 3.12, PyTorch, Triton, FastAPI and gRPC. Exact
dependencies and the independent vLLM submodule are pinned in `uv.lock` and
`versions.lock.yaml`.

## What actually runs

| Layer | Current validation |
|---|---|
| Managed vLLM Worker | Real Qwen2.5-0.5B weights on workstation A5000/A4000: Chat Completions, Responses and Anthropic Messages, each with JSON, SSE and cancellation after generated text; duplicate admission rejected |
| Token preparation | Native three-protocol rendering on A5000/A4000 agrees with response usage; request/body/tenant expiry, actual engine-input and block-rounded reservation checks precede execution |
| KV geometry | Allocator-reported block counts and per-block bytes verified on A5000/A4000; excludes the null block and reports gross capacity separately from Scheduler reservations |
| Execution boundary | Durable route identity aggregates native engine calls; only EngineCore removal plus CUDA synchronization permits a terminal receipt |
| Gateway/Scheduler | Native preparation is connected to per-worker resource reservations through gRPC; CPU integration contracts pass. Automatic GPU bootstrap is not yet accepted |
| Complete deployment | Worker registration/heartbeat, gross-budget accounting and full Gateway→Scheduler→Worker GPU bootstrap remain incomplete |
| Hardware expansion | A6000 validation is pending; 3×4 H100 is a future hardware target, not a verified deployment |

The managed execution adapter currently accepts synchronous TP=PP=DP=1 text
inference with `n=1`, without KV/EC transfer connectors or background Responses.
Unsupported modes fail explicitly. This is an implementation slice, not a reduction
of the final plan. There is no claimed Agent-task latency or throughput gain.

## Run the managed Worker

Use the pinned fork container built as described in `docs/operations.md`.
Keep `FREECHAT_WORKER_TOKEN` in your environment/secret manager (at least 32
characters); never commit it. The same token is configured on the Gateway and Scheduler.
Build the managed image from this checkout, then resolve its image ID:

```bash
docker build -f deploy/worker/Dockerfile \
  --build-arg WORKER_BASE=freechat-worker:FORK_SHA_FROM_VERSIONS_LOCK \
  -t freechat-worker:development .
docker image inspect freechat-worker:development --format '{{.Id}}'
docker run --name freechat-worker --gpus all --shm-size 1g \
  -e FREECHAT_WORKER_TOKEN \
  -p 127.0.0.1:8000:8000 \
  -v /absolute/path/to/Qwen2.5-0.5B-Instruct:/model:ro \
  -v freechat-worker-state:/var/lib/freechat \
  IMAGE_ID_FROM_INSPECT \
  --worker-id gpu-worker --model /model --served-model-name qwen \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.2 --max-model-len 2048 --enforce-eager
```

The container provides the GPU dependencies; CPU `uv sync` alone does not install
vLLM or make a working GPU environment. The underlying entry point is
`python -m freechat_worker.serve` (`freechat-worker` when installed as a package).

The service uses vLLM's original protocol handlers, not protocol rewrites. Its
execution gRPC endpoint is `127.0.0.1:50052`. The state directory contains the
normal service admission journal and owner lock, **not validation results**.
Do not reuse a generation with a different engine incarnation. The default
generation changes on startup; a restart is not automatically proof that all
old Scheduler reservations have been reconciled.

Container packaging is in `deploy/worker/Dockerfile`; provide the pinned fork
base image using `--build-arg WORKER_BASE=...`. Resolve the built image ID before
testing. Build instructions and release-image limits are in `docs/operations.md`.

## Validate, with no result directories

Against the running Worker, execute the GPU validator inside its container:

```bash
docker exec freechat-worker /opt/freechat/.venv/bin/python \
  -m tools.validate_native_http --model qwen
```

Run the partial CPU checks from the source checkout:

```bash
uv sync --all-packages --group dev
uv run pytest -m "not gpu and not multinode"
uv run ruff check .
uv run mypy libs services worker benchmarks tools
```

The first command invokes real HTTP/SSE and execution RPC on the loaded GPU.
CPU tests check only their local logic/contracts; they do not replace GPU,
multi-node, real-Harness or performance tests. Validators print results to stdout;
services emit operational logs. Benchmark raw records are hash-linked JSON log
records, not generated result directories. Profiler output can be streamed with
`--chrome-trace`. Existing historical evidence is retained, not regenerated.

The Compose control services and five-page WebUI exist, but their startup is not
yet a complete inference bootstrap. Follow the root plan's remaining integration
work rather than bypassing the Scheduler with a static Worker URL.

## One development route

All edits happen on ross, `main`, in
`/home/linkst/workspace/projects/free-chat`. Workstation runs artifacts built from
ross; it is not a second source checkout. Validated feature branches, when needed,
must merge back to `main`.

Initialize `third_party/vllm` with `git submodule update --init --recursive`.
The gitlink and version lock must agree; do not use `submodule update --remote`.
Run `uv run python -m tools.check_source_checkout` to check source consistency.
The current fork remote is hosted on ross, so another machine needs SSH access to
that repository or a deliberately configured published mirror.

There is one authoritative plan: `agent-aware-inference-infra-plan.md`.
Operational instructions, measurement methods and claim boundaries live in
`docs/operations.md`, `docs/benchmark-metrics.md`, and `docs/claims-ledger.md`.
Do not delete the plan until its acceptance gates pass and the owner approves.
