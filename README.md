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
| Gateway/Scheduler | Real same-host Qwen2.5-0.5B inference on A5000/A4000 through automatic registration, authenticated heartbeat, native preparation, reservations and Worker execution reconciliation |
| Concurrent admission | Three 8-request contention rounds per GPU against an actual 128-block pool on A5000/A4000; capacity backpressure, re-admission and complete fenced release verified, with no pool oversubscription |
| Shared Scheduler | A5000/A4000 execute real concurrent requests under one Scheduler; per-GPU capacity and fenced release pass independent log audits. After an idle A4000 stops gracefully, new requests execute on A5000 |
| Scheduler restart | On A5000 with real single-node etcd, killing Scheduler after first generated token retains the reservation; restart reconciles the same live Worker/engine and admits new work |
| Lifecycle delivery | Real A5000 inference and durable events pass short and 150-second NATS outage tests; Scheduler recovers delivery and admits new GPU work without restarting. Publish/consumer acknowledgment boundaries remain at-least-once |
| SDK function tools | Real OpenAI Agents SDK on A5000/A4000 executes README tool and Active/Resume requests through the managed loop; both strict answer-format checks fail. Request execution/release is verified, task-quality acceptance is not |
| Worker restart | A5000 operator-assisted recovery reclaims sealed old-incarnation reservations before replacement; A4000 also verifies independent historical receipts after the replacement registers and serves new work. Automatic failover and original-task resumption remain incomplete |
| Complete deployment | Same-host API paths, Scheduler restart and short/sustained NATS outages are exercised; Worker-crash recovery, store faults, consumer recovery, WebUI integration and cross-node deployment remain incomplete |
| Hardware expansion | A6000 validation is pending; 3×4 H100 is a future hardware target, not a verified deployment |

The managed execution adapter currently accepts synchronous TP=PP=DP=1 text
inference with `n=1`, without KV/EC transfer connectors or background Responses.
Unsupported modes fail explicitly. This is an implementation slice, not a reduction
of the final plan. There is no claimed Agent-task latency or throughput gain.

## Run the same-host inference loop

Use the pinned fork container built as described in `docs/operations.md`.
Keep `FREECHAT_WORKER_TOKEN` in your environment/secret manager (at least 32
characters); never commit it. Configure `FREECHAT_CACHE_SALT_SECRET` (at least 32
bytes) and `FREECHAT_VALIDATION_API_KEY` separately. These are development service
secrets, not client-controlled hints. Build both images from this checkout:

```bash
docker build -f deploy/worker/Dockerfile \
  --build-arg WORKER_BASE=freechat-worker:FORK_SHA_FROM_VERSIONS_LOCK \
  -t freechat-worker:development .
docker build -f deploy/images/control.Dockerfile -t freechat-control:development .
export WORKER_IMAGE="$(docker image inspect freechat-worker:development --format '{{.Id}}')"
export CONTROL_IMAGE="$(docker image inspect freechat-control:development --format '{{.Id}}')"
export FREECHAT_API_KEYS="validation:${FREECHAT_VALIDATION_API_KEY}"
export FREECHAT_RUNTIME_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"

docker run -d --name freechat-worker --gpus device=0 --shm-size 1g \
  --label "io.freechat.runtime-id=$FREECHAT_RUNTIME_ID" -e FREECHAT_RUNTIME_ID \
  -e FREECHAT_WORKER_TOKEN -e FREECHAT_VALIDATION_API_KEY \
  -p 127.0.0.1:8080:8080 \
  -v /absolute/path/to/Qwen2.5-0.5B-Instruct:/model:ro \
  -v freechat-worker-state:/var/lib/freechat \
  "$WORKER_IMAGE" \
  --worker-id gpu-worker --model /model --served-model-name qwen \
  --host 127.0.0.1 --port 8000 \
  --scheduler-target 127.0.0.1:50051 --node-id local-node \
  --gpu-memory-utilization 0.2 --max-model-len 2048 --enforce-eager

docker run -d --name freechat-scheduler --network container:freechat-worker \
  -e FREECHAT_WORKER_TOKEN -e FREECHAT_SCHEDULER_LISTEN=127.0.0.1:50051 \
  "$CONTROL_IMAGE" freechat-scheduler
docker run -d --name freechat-gateway --network container:freechat-worker \
  -e FREECHAT_WORKER_TOKEN -e FREECHAT_API_KEYS -e FREECHAT_CACHE_SALT_SECRET \
  -e FREECHAT_SCHEDULER_TARGET=127.0.0.1:50051 \
  -e FREECHAT_ORIGIN_NODE_ID=local-node \
  "$CONTROL_IMAGE" freechat-gateway
```

The container provides the GPU dependencies; CPU `uv sync` alone does not install
vLLM or make a working GPU environment. The underlying entry point is
`python -m freechat_worker.serve` (`freechat-worker` when installed as a package).

The three containers deliberately share the Worker's network namespace: only the
Gateway is published on host loopback port 8080. This is a same-host launch path,
not a cross-node network workaround. The Worker registers after engine/HTTP
readiness and retries with measured heartbeats. Do not submit direct Worker tests
concurrently with Scheduler-owned work.

The service uses vLLM's original protocol handlers, not protocol rewrites. Its
execution gRPC endpoint is `127.0.0.1:50052`. The state directory contains the
normal service admission journal and owner lock, **not validation results**.
Do not reuse a generation with a different engine incarnation. The default
generation changes on startup; a restart is not automatically proof that all
old Scheduler reservations have been reconciled. Each newly created Worker container
needs its own runtime ID in both its Docker label and environment; reuse that ID
only when restarting the same container. Shared-network hostname is not container
identity. The bounded, operator-assisted retirement procedure is in `docs/operations.md`.

Container packaging is in `deploy/worker/Dockerfile`; provide the pinned fork
base image using `--build-arg WORKER_BASE=...`. Resolve the built image ID before
testing. Build instructions and release-image limits are in `docs/operations.md`.

## Validate, with no result directories

Against the running three-service loop, execute the GPU validator. For two GPUs under
one Scheduler, use the peer launch and per-worker log audits in `docs/operations.md`;
wait for both current Worker generations to register before starting contention:

```bash
docker exec freechat-worker /opt/freechat/.venv/bin/python \
  -m tools.validate_inference_loop --model qwen
```

Run the partial CPU checks from the source checkout:

```bash
uv sync --all-packages --group dev
uv run pytest -m "not gpu and not multinode"
uv run ruff check .
uv run mypy libs services worker benchmarks tools
```

The first command exercises all three Gateway protocols, generated-text streaming
and disconnects, then sequentially reserves more than one physical KV pool to
check capacity reuse. Correlate its decision IDs with Scheduler `lifecycle_event`
logs: cancellation intent and terminal, quiescent, admission-closed Worker receipts
must precede release. `tools.validate_native_http` remains the isolated Worker
probe, not a replacement for this loop.
CPU tests check only their local logic/contracts; they do not replace GPU,
multi-node, real-Harness or performance tests. Validators print results to stdout;
services emit operational logs. Benchmark raw records are hash-linked JSON log
records, not generated result directories. Profiler output can be streamed with
`--chrome-trace`. Existing historical evidence is retained, not regenerated.

For concurrent admission correctness, launch an isolated Worker with
`--num-gpu-blocks-override 128 --max-model-len 1024`, then use
`tools.validate_inference_loop --mode concurrent`. This changes the actual vLLM
allocation, not the reported budget. The probe must observe successful requests,
capacity rejection and subsequent admission; its stdout supplies counts for the
normal-log audit in `docs/operations.md`. A bounded retry accommodates asynchronous
release confirmation. This is not a throughput benchmark.

The default local Scheduler uses in-memory state and normal lifecycle logs unless
etcd/NATS are configured; this is not restart recovery or HA evidence. Compose
and the five-page WebUI still need integration acceptance. No measured Agent-task
performance claim follows from sequential capacity reuse.

Stop the development loop when finished; state and logs remain available:

```bash
docker stop freechat-gateway
docker stop freechat-worker
docker stop freechat-scheduler
```

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
