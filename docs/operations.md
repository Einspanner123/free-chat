# FreeChat 操作手册

唯一实施顺序见根目录计划，当前入口见 README。验证只输出到 stdout 或正常服务日志，不创建结果目录。
服务 journal、模型缓存和配置不是验证结果，按正常服务生命周期管理。

## Managed Worker

`freechat-worker` 使用原生 vLLM 三协议处理器，外层只负责认证、准入和执行关联。
启动命令见 README；控制 RPC 当前只绑定同机 loopback，不宣称跨节点 mTLS 已完成。
HTTP 除 `/health` 外要求 `x-freechat-worker-token`；Gateway 从服务配置提供该 token，
不会转发客户端提供的同名身份。租户和执行身份由 Gateway/调度结果生成。

部署镜像使用 `deploy/worker/Dockerfile`。运行后在同一主机或容器内执行：

```bash
python -m tools.validate_native_http --model qwen
```

该命令覆盖原生三协议的普通响应、SSE、生成中取消、重复准入拒绝和未认证请求拒绝。
它不代替尚未接通的完整 Gateway/Scheduler 启动与容量对账。
缓存事件通过 `freechat_worker.benchmark_hook:create_cache_log_hook` 写入服务 logger，
消息前缀 `FREECHAT_CACHE_EVENT`；不再创建独立 JSONL 文件。

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
  --gpu 1
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

## 请求准备与执行预算

Managed Worker 的推理入口要求 `x-freechat-internal-preparation`。先向认证的
`POST /freechat/prepare` 提交 `{"protocol":"/v1/responses","request":原生请求对象}`，
并提供由 Gateway 认证身份产生的 `x-freechat-internal-tenant`。响应包含 preparation ID、
原生渲染后的 input/output token 预算以及 Worker generation/engine identity。
随后提交同一原生请求体、execution identity 和 `x-freechat-internal-reserved-kv-bytes`；
Worker 用实际 block geometry 检查预占字节恰好覆盖 input/output token，且不超过物理池与上下文。
只有调度生成的 `agent_lifecycle` 元数据
允许在准备后添加。预算只在同一 Worker incarnation 内有效，默认 60 秒、最多 4096 条。

处理器复用 vLLM 的请求校验、模板渲染和输出 token 上限计算，不复制模板或估计字符数。
上游解析器处理独立副本，避免原地补默认字段改变原请求哈希。提交 GPU 前再次核对
实际 prompt token 指纹和采样上限；不一致、过期或额外内部模型调用均明确拒绝。
当前 native built-in tool loop 仍需逐调用准入，不等于外部 Harness 的多轮工具调用已禁用。

`tools.validate_native_http` 自动执行准备，按实测 block geometry 提供测试预算，核对真实
JSON response usage，并验证 SSE、取消与重复拒绝。这是直接 Worker 验证，不声称获得了 Scheduler 租约。

Gateway 已移除字符数预算；它通过 gRPC 暂态传送经过身份处理的原生请求。
Scheduler 先过滤模型、健康度和远端使用权限，再向各候选 Worker 请求原生预处理，
分别计算成本和逐 rank 预占；CAS 重试时重查 generation、engine identity、请求绑定与有效期。
prepare 不可用时拒绝，不退回字符估算。原生请求不进入 ledger/event/repr，持久化指纹与预算元数据。
默认 Scheduler 入口必须配置 `FREECHAT_WORKER_TOKEN`；`contract_only=True` 仅用于显式 CPU fixture。
当前 managed 路径无 KV connector，因此 offload 指令明确标为未启用，不在准备后改写请求。

节点时钟必须同步，过期/未来 telemetry 不可绕过。自动注册、持续 heartbeat、gross budget 与
活动请求预占不重复计数仍待接通，不能将这些 CPU 集成测试和单 Worker GPU 测试合称为全链路验收。

## KV 容量报告

Managed Worker 在分配 KV 后调用 vLLM 原生 Worker extension 的具名 RPC，
读取逐 rank 的 block 数、page bytes 和 tensor 配置；不启用不安全函数序列化。
`/freechat/runtime` 的认证响应包含 `capacity`。当前支持单 rank、单 full-attention/MLA
缓存组；混合、不明或不一致布局拒绝启动。`basis` 明确为 gross engine pool，
其中一个 null block 不可分配。它既不是 CUDA 空闲显存，也不是扣过请求预占的余额；
不得直接与已扣预占的实时空闲计数混用。目前还未接入 Scheduler 的真实准入闭环。

## Telemetry、校准与 Harness 测试

- `python -m freechat_worker.telemetry --help`：采集原生 Prometheus 队列、KV 使用率与传输计数，发送带 generation/engine identity 的 heartbeat。先注册同一 capability/generation。采集器目前不提供可信的 KV 空闲字节预算或完整 prefix inventory。
- `python -m benchmarks.calibrate_service --help`：Prefill/Decode 服务时间校准；每份 profile 绑定模型、Worker generation、engine instance 与观测时间。
- `python -m benchmarks.calibrate_transfer --help`：完整 block 的 Store/Load 测量，检查 token/字节一致性；传输服务率不等于端到端吞吐。
- `python -m benchmarks.harness_offload_boundary --help`：独立 baseline Worker 上的 Harness 边界实验，不是已完成的 managed Gateway 路由验收。
- `python -m benchmarks.analyze_offload_boundary`：从 stdin 读取上述 hash-linked JSONL；拒绝重复/损坏记录，结果输出到 stdout。可传入历史目录做只读分析。
- `python -m benchmarks.profile_kv_quantize --help`：GPU profiler；`--chrome-trace` 将 trace 写到 stdout，摘要写 stderr，不生成结果目录。

这些校准程序需按其 CLI 指定实际模型、capability 和运行环境。不要把直接访问 baseline Worker 的结果当作 Gateway/Scheduler 收益。历史测量和 claim 边界仅在 Claims Ledger 维护，操作手册不保留逐次试验过程。

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
Only a real Harness process, real Gateway/worker request, service trace and
protocol-matched failure matrix may satisfy `harness-integrations` in the
Claims Ledger. No event-contract result is a latency, cache-hit or throughput
claim.
