# FreeChat 操作手册

实施顺序和未完成项只看根目录 `agent-aware-inference-infra-plan.md`。
本文仅提供运行/采集接口，不表示每个步骤已完成端到端验收。

## 当前执行后端 GPU 验证

源码在 ross 修改，测试镜像可在 GPU 主机从 ross 传入的构建上下文构建。
`deploy/worker/Dockerfile.execution-test` 复用已有锁定镜像的二进制并覆盖本次 Python 源码，
执行 `tools.validate_execution_gpu`：真实权重、多次完成、提交后/首 token 后取消和重复准入拒绝。
只支持同步单 GPU、文本 n=1，无 disaggregated KV/EC；不是完整 HTTP serving launcher。
运行时挂载模型只读，并记录基础/派生镜像 ID、父仓库/fork revision 及原始日志。
不可将该开发镜像标为 release，或将这个 probe 标为整个系统测试通过。

GPU 主机构建后先用 `docker image inspect` 解析镜像 ID，再运行固定 ID：

```bash
docker run --rm --init --gpus device=0 --shm-size=2g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /data/freechat/models/Qwen2.5-0.5B-Instruct:/model:ro \
  sha256:IMAGE_ID --model /model --repeats 12
```

Gateway 运行必须配置 `FREECHAT_SCHEDULER_TARGET`；缺失或空白直接拒绝启动。


## Worker container build and acceptance

The worker container is the only accepted CUDA compiler and runtime boundary.
Host `nvcc`, Python packages and profiler binaries are not inherited as evidence.

### Build from the pinned fork

Build from the root of the vLLM fork at the commit in `versions.lock.yaml`:

```bash
docker buildx build \
  --file docker/Dockerfile \
  --target vllm-openai \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg torch_cuda_arch_list=8.6 \
  --build-arg VLLM_USE_PRECOMPILED=1 \
  --build-arg VLLM_MERGE_BASE_COMMIT=9c22668436a4d94aab87ea74a220e060415cf1d8 \
  --build-arg VLLM_MAIN_CUDA_VERSION=13.0 \
  --label org.opencontainers.image.revision=FORK_SHA_FROM_VERSIONS_LOCK \
  --label io.freechat.vllm.upstream-revision=9c22668436a4d94aab87ea74a220e060415cf1d8 \
  --tag freechat-worker:FORK_SHA_FROM_VERSIONS_LOCK \
  --load \
  .
```

The fork changes Python lifecycle, scheduling and cache-policy integration but
does not modify vLLM C++ or CUDA sources. The build therefore reuses the
commit-specific CUDA 13 extension wheel published for the exact upstream
commit recorded in `versions.lock.yaml`, while packaging the fork's Python and
Rust layers from source. Both revisions are OCI labels and acceptance checks;
using a wheel from a different upstream commit fails closed. A source build
remains available for auditing, but is not required merely to rebuild unchanged
upstream kernels.

The build is not release evidence. Push it to the project registry, resolve the
repository digest, and replace `worker_image_digest: UNRESOLVED` only after the
GPU acceptance command succeeds against the digest reference.

### Fail-closed acceptance

```bash
uv run python -m tools.validate_worker_image \
  registry.example/freechat-worker@sha256:REPLACE_ME \
  --gpu 1 \
  --output /data/freechat/profiles/worker-image.json
```

Acceptance requires all of the following in one run:

- Python 3.12, PyTorch 2.13.0+cu130, CUDA runtime and `nvcc` 13.0,
  Triton 3.7.1 and Transformers 5.16.1.
- A CUDA-visible physical GPU and a real FlashInfer top-k/top-p sampling kernel
  invocation, including synchronization so JIT/compiler failures are observable.
- The authoritative fork revision in the OCI image label.
- The exact upstream revision that supplied the precompiled CUDA extensions.
- An immutable repository digest rather than a mutable local tag.

The upstream `vllm/vllm-openai:v0.26.0` image is retained only as a direct-vLLM
baseline. Its PyTorch 2.11 and Triton 3.6 packages do not satisfy the candidate
worker lock.

### Host prerequisite

Docker must expose the NVIDIA runtime before the GPU gate can run. Installing or
reconfiguring NVIDIA Container Toolkit and restarting Docker are host operations;
they must be scheduled so unrelated running containers are not interrupted. A
CPU-only container probe or a successful image pull does not satisfy this gate.
Omitting `--gpu` always produces a non-accepted inspection report.

The local candidate passed every GPU and locked-runtime check on an A4000 on
2026-09-07, including synchronized FlashInfer sampling. It remains unaccepted
because no project registry is configured and the local tag has no repository
digest. A registry push and a repeat of the same command against the resolved
digest are still required; the local image ID is not a substitute.

## Worker telemetry and calibration boundary

The collector uses the pinned engine's Prometheus endpoint and host GPU memory
observation, then emits generation-fenced gRPC heartbeats. It supports one model
and engine `0` per worker. It selects exact metric names and labels; unrelated
multi-dimensional series are ignored, while ambiguous selected series fail.

Run `python -m freechat_worker.telemetry --help` on the GPU host. Register the
worker first and supply the same capabilities file and generation. The engine
instance identifier must change on engine restart, along with worker generation.
The collector stops on scrape failure or reset; existing telemetry ages out.
Scheduler rejects samples older than 30 seconds or more than five seconds in
the future. Heartbeats cannot replace newer observations or change engine
instance within an existing generation. Internal transport authentication is
still a separate deployment gate; these checks are consistency checks.

### Available observations

- Running and waiting request counts, plus physical free VRAM from nvidia-smi.
- Engine KV usage ratio. The pinned BlockPool computes this as one minus free
  blocks divided by total blocks excluding its null block. It does not expose
  the identity/value of reusable cached prefixes. Consequently the collector
  leaves KV capacity/free-byte and prefix inventory fields unknown.
- Store/load bytes divided by accumulated operation duration, calculated from
  consecutive scrapes in the same engine instance. This is transfer service
  throughput, not end-to-end request speed or wall-clock PCIe saturation.
- Windows without transfer observations emit no store estimate. A first scrape
  establishes the baseline; missing counters are not silently treated as zero.

### Remaining calibration work

Prefill/decode throughput, KV layout-derived capacity and reusable block lineage
are not yet calibrated by this collector. Existing numeric contract defaults
are not measurements. Scheduler disables predictive offload for this source
with `prefill_calibration_required`; route cost estimates still use contract
defaults and must not be used in performance claims or heterogeneous-placement
acceptance. The next change must replace those defaults with calibrated profiles
and an explicit unavailable-cost fallback.

### Live validation

`benchmarks/telemetry_probe.py` connects real worker HTTP metrics to an isolated
in-memory scheduler over gRPC. It sends a direct real-model request with explicit
native offload enabled to produce a transfer measurement; this is not a policy
performance experiment. Run it with a registered-model capabilities file and
a unique evidence output directory. At least one prior transfer is required to
initialize the native engine's lazily emitted counters.

On 2026-09-08, A5000 validation `evidence/telemetry/20260908-run03/` accepted both
heartbeats, observed a store-window service rate of approximately 11.112 GB/s,
and returned `prefill_calibration_required` through the real gRPC route client.
This one window provides no confidence interval and does not establish expected
bandwidth for other shapes. The raw before/after metrics and hashes are retained.

Run01 retained a failed parser probe (unrelated multi-label metrics); run02
retained the first-counter initialization case. Run03 is the successful probe.
No run is a Harness or end-to-end performance acceptance result.

## Scoped inference service calibration

The Scheduler consumes measured service profiles instead of interpreting telemetry's
legacy default rates as measured inference costs. Profiles bind worker generation,
engine instance, model capability, observation/expiry timestamps, image identity and
an observation artifact hash. Matching requires the measured input/output scope and
concurrency scope; stale, missing or mismatched profiles are unavailable estimates.
If any eligible candidate lacks a usable estimate, cost-aware and lifecycle-aware
routing explicitly fall back to least-load. The decision carries requested strategy,
effective strategy, fallback reason and per-candidate estimate availability.

Numeric zero fields in an unavailable candidate are serialization placeholders, not
zero-latency predictions. Consumers must inspect `estimate_available`. Queue, network
and eviction terms are not comprehensively calibrated by this work.

### Real execution, 2026-09-08

`evidence/calibration/20260908-run01/` contains raw before/after Prometheus snapshots,
response usage, observations, scoped profiles, gRPC routing decisions and a SHA-256
manifest. The run used workstation A5000 and Qwen2.5-0.5B-Instruct, with three measured
serial requests per scope after warm-up. Both scopes generate exactly 16 tokens:

| Input tokens | Prefill tokens / engine service second | Decode tokens / engine service second |
| --- | ---: | ---: |
| 254 | 31505.08 | 420.56 |
| 926 | 54923.61 | 413.76 |

These are pooled calibration rates, not task throughput, speedup, TTFT or an SLO.
Decode counts output tokens minus the first token against the engine decode interval.
The collector requires one completed request in every histogram, idle boundaries,
unchanged preemption count, positive deltas, response/metric token agreement, and
computed KV tokens equal to prompt tokens. Unique cache salts prevent prefix reuse.
The actual image's metric definition excludes cached tokens from computed KV tokens.

The probe validated profiles through real gRPC heartbeats and Gateway routing-client
serialization against an isolated in-memory Scheduler. In-scope requests received a
calibration ID. An out-of-scope request received least-load plus
`candidate_cost_unavailable`. This does not validate the deployed persistent control
plane. Profiles expire after one hour and are not portable across engine restarts.

### Remaining gates

- The image identity is a local image SHA, not a registry digest. The model/tokenizer
  revision labels are not an independently verified complete artifact manifest.
- Two exact input lengths with three samples each do not establish interpolation,
  confidence intervals, concurrent batching or cached-prefix service costs.
- Store/Load size-scoped calibration is absent. Predictive offload remains disabled
  with `transfer_calibration_required`, even when inference calibration matches.
- KV residency and eviction probability remain separate work; free VRAM and cache
  usage do not establish prefix residency or future reuse.
- Four-Harness paired task-level comparisons remain uncompleted. No resume benefit
  percentage may be derived from these service rates.

Continue with Store/Load calibration and residency evidence, then evaluate predictive
offload using paired real Harness trials. The post-tool-wait fork interface remains
deferred under the recorded A-before-B decision. Keep the implementation plan.

## Native KV transfer calibration evidence

The executable probe is `python -m benchmarks.calibrate_transfer`. It uses an
already-running, dedicated single-model worker. The pressure recipe is specific to
the A5000 probe container's 64 MiB KV pool, not general topology discovery.

Each scope runs a warm-up cycle followed by three measured cycles:

1. Submit a new salted target with native prompt offload enabled.
2. Submit eight independently salted pressure requests with offload disabled.
3. Resume the exact target with the same salt and offload disabled.
4. Compare isolated request windows for Store/Load byte, operation and service-time
   deltas. Require external prefix hit tokens in every accepted Load window.

The observer rejects missing counters, counter regression/disappearance, non-idle
boundaries, preemption, multiple completed requests, zero transfer amounts and
non-integral accounting. This is a controlled experiment: boundary checks alone do
not establish exclusive access against arbitrary concurrent clients.

### Evidence

`evidence/transfer-calibration/20260908-run01/` records two successful scopes, each
with three Store and three Load observations after warm-up, on real A5000 inference:

| Scope | Bytes per Store and Load | External hit tokens per resume |
| --- | ---: | ---: |
| 32 repetitions | 2,949,120 | 240 |
| 128 repetitions | 11,206,656 | 912 |

These values match 12,288 KV bytes/token for the configured Dense/GQA model. They
are complete-block transfers, not all prompt tokens. Service-time ratios appear in
the raw observations only as calibration data, not task throughput or speedup.

The successful follow-up `evidence/transfer-calibration/20260908-run02/` also archives
target/resume responses and its source hash, and
requires equal target/resume prompt counts plus Load bytes equal to external hit
tokens times the configured KV layout size. Each run has a separate output directory;
raw earlier evidence is retained, never overwritten.

### What this does not establish

- No real Harness or baseline/candidate performance comparison was run here.
- The two exact byte sizes do not justify interpolation, concurrent-copy cost or
  a universal bandwidth value. Three samples do not establish a stable tail estimate.
- The aggregate external-hit counter is not a per-block residency catalog. The
  current KV controller policy and worker callback buffer do not supply a complete,
  reconciled native CPU/GPU residency inventory. Dropped events must invalidate
  certainty before any catalog is used for routing.
- Image identity and engine instance are probe arguments; production attestation
  and full model/tokenizer artifact identity remain separate gates.
- These measurements are not automatically merged into Scheduler profiles. The
  current request-size estimate uses total prompt tokens, while the native connector
  transfers complete blocks. Calibrate the actual transfer plan and its direction,
  size and generation before authorizing predictive offload from these samples.

Next: implement the explicit block-aligned transfer-plan contract, connect native
completion/residency signals with uncertainty handling, then run paired real Harness
trials. The A-before-B decision and plan-retention requirement remain unchanged.

## Explicit future-reuse forecasts

The four Harness adapters no longer assign 0.9 to every growing-history request.
Lifecycle identity is an observation, not a calibrated probability of future reuse.
Without an applicable forecast the adapter emits probability zero and metadata
`reuse_forecast_status=unavailable`. Zero here is the conservative control value,
not an empirical prediction that reuse is impossible; exclude unavailable values
from calibration scoring and report their coverage separately.

`HarnessCall.reuse_forecast` accepts a `ReuseForecast` containing task, session,
agent, branch and call identity; probability; future reuse horizon in milliseconds;
timezone-aware observation and expiry; and an evidence reference. Validity is capped
at one hour. Expired, future-dated or differently scoped forecasts are not applied.
Terminal/cancelled calls never authorize future copies through the adapter.

The evidence reference is a caller provenance label, not a certificate that a model
has been calibrated. The adapter does not authenticate it against a training artifact
registry. Such validation and actual forecast production remain required work.

The Scheduler rechecks adapter forecast identity and validity after transport, so
a forecast that expires between hint construction and routing cannot authorize
offload. Unavailable or malformed adapter forecasts return explicit refusal reasons.
Existing direct clients without forecast metadata retain the explicit-hints protocol;
this compatibility path is not upgraded to calibrated evidence by this change.
Do not call this interface a trained predictor or a performance improvement.

When a forecast applies, `expected_resume_ms` carries its future horizon and metadata
contains its reference, call and timestamps. When it does not apply, the existing
Tool Wait/Resume duration field remains available for lifecycle compatibility, but
the reuse probability is zero. A past tool duration alone cannot authorize offload.

Predictions must be constructed for the actual next model call identity. Replacing
the call, agent or branch does not transfer the prediction. The SDK-specific runtime
hooks do not yet fetch forecasts from a producer automatically; this is the shared
edge contract and safe default, not completed live predictor integration.

### Metric completion requirements

- Log each forecast before dispatch, with exact scope, observation time, expiry,
  versioned producer identity and validated evidence reference.
- Join it to the later outcome by authenticated task/call identity, not position.
- Record reuse and expiry/terminal closure; pending outcomes are censored, not false.
- Score only valid, resolved predictions with Brier score, calibration bins and
  coverage by Harness/workload. Report missing and invalid forecast counts separately.
- Charge unused Store bytes and repeated Prefill from false negatives separately;
  a well-calibrated reuse probability does not itself establish an eviction probability.
- Use held-out real Harness traces; never fit and validate on the same paired probe.

Current evidence is CPU contract tests across all four adapters. No new GPU or
end-to-end improvement claim is introduced by this change. The A implementation route
and original implementation-plan retention requirements are unchanged.

## Harness adapter boundary

FreeChat does not own an Agent loop. Harness adapters translate stable runtime
identity and lifecycle events into the common `freechat.agent_hints` request
extension; the Gateway authenticates the tenant and validates the extension,
and the Scheduler retains final resource authority.

### Supported contract depth

| Harness | Implemented boundary | Verified scope | Remaining acceptance |
|---|---|---|---|
| OpenAI Agents SDK | Run hooks plus model decorator | Real `Runner.run()` tool call traversed the Gateway and Qwen2.5-0.5B Worker; the second model request carried Resume identity | Cancellation/failure replay, task-quality gate and benchmark matrix |
| LangGraph | Runnable/checkpoint context bridge | Real interrupt and `Command(resume=...)` preserve task, thread and checkpoint identity | Gateway, real model, branching/failure replay and benchmark matrix |
| OpenCode | Native session-event state machine | ToolPart pending/running/completed/error, parallel calls, replay regression and cross-session rejection | Live plugin/SSE run, model request interception, cancellation/failure replay and benchmark matrix |
| OpenHands | SDK event state machine | Action/Observation/error pairing, parallel calls and late-event replay protection | Live SDK callback, model request interception, cancellation/failure replay and benchmark matrix |

The two event bridges are dependency-light on purpose. They accept native event
dictionaries at the transport boundary and do not import private framework
internals. This makes recorded trace replay deterministic and keeps framework
upgrades from entering the Scheduler contract. It does not substitute for live
Harness acceptance.

### OpenCode bridge

Create one bridge for one session, feed it `message.part.updated` and session
events, then apply the current state immediately before the model request:

```python
from freechat_harness_adapters import OpenCodeLifecycle

bridge = OpenCodeLifecycle(session_id="ses_123", task_id="task_123")
bridge.on_event(event)
request_body = bridge.apply(request_body)
```

The bridge keys tool state by `callID` and records `messageID` as the turn.
Pending and running calls produce Tool Wait. Completed and failed calls produce
Resume only after every parallel call is terminal. A late pending/running update
cannot reopen a completed call.

### OpenHands bridge

Create one bridge for one conversation and feed events from the conversation
callback before applying hints to the next model request:

```python
from freechat_harness_adapters import OpenHandsLifecycle

bridge = OpenHandsLifecycle(conversation_id="conversation-123", task_id="task-123")
bridge.on_event(event)
request_body = bridge.apply(request_body)
```

Action events enter Tool Wait. Observation, user-rejection and agent-error
events close their matching `tool_call_id`; Resume begins only when no parallel
call remains. Conversation errors and interrupts enter Cancelled. Terminal and
Cancelled states are sticky so delayed replay cannot resurrect a task.

### Evidence rule

Unit or contract tests may verify identity preservation and transition logic.
Only a real Harness process, real Gateway/worker request, archived trace and
protocol-matched failure matrix may satisfy `harness-integrations` in the
Claims Ledger. No event-contract result is a latency, cache-hit or throughput
claim.

The 2026-09-07 OpenAI Agents run executed `read_file("README.md")` and recorded
paired Active and Resume model requests followed by Terminal. The 0.5B model's
final answer did not match the expected README heading, so this run validates
the lifecycle path but explicitly fails the task-quality gate. The archived
record is `evidence/harnesses/openai-agents-qwen05b-20260907.json`; it must not
be used as performance or model-routing acceptance.
