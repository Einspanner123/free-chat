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
它不代替 Gateway/Scheduler 启动与容量对账；完整同机入口及 `tools.validate_inference_loop` 见 README。
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

节点时钟必须同步，过期/未来 telemetry 不可绕过。默认 Worker 通过 `--scheduler-target` 和
`--node-id` 启用自动注册、持续 heartbeat；注册地址、HTTP 和执行 RPC 目前只接受同机 loopback。
Gateway/Scheduler/Worker 在同一个网络 namespace 中运行，不通过放开 peer 校验冒充跨节点部署。
同机 API 推理闭环已在 A5000/A4000 分别运行；多 Worker、跨节点和重启恢复仍待验收。

## KV 容量报告

Managed Worker 在分配 KV 后调用 vLLM 原生 Worker extension 的具名 RPC，
读取逐 rank 的 block 数、page bytes 和 tensor 配置；不启用不安全函数序列化。
`/freechat/runtime` 的认证响应包含 `capacity`。当前支持单 rank、单 full-attention/MLA
缓存组；混合、不明或不一致布局拒绝启动。`basis` 明确为 gross engine pool，
其中一个 null block 不可分配。它既不是 CUDA 空闲显存，也不是扣过请求预占的余额；
不得直接与已扣预占的实时空闲计数混用。

Managed heartbeat 明确标记 `scheduler_exclusive_gross`，Scheduler 从该物理池扣减自身请求预占，
活动请求用观测值与预占数的最大值而非相加，避免同一请求重复计数。此模式要求请求经过同一
Scheduler，禁止同时直接向 Worker 压测；`unmanaged_observation` 保留独立观测的原有语义。
GPU UUID、KV geometry 和 CUDA free memory 来自原生 Worker extension；队列/活动数来自原生指标。
注册的模型 revision 是只读本地权重和 tokenizer 文件内容清单的 SHA-256，不冒称上游仓库 commit。
健康检查失败发布 unhealthy；控制面不可达时已有观测自然过期。没有真实预算不准入。

## 生命周期清理与运行日志

默认 Scheduler 根据注册的 execution endpoint 轮询 Worker，只接受同一 generation/engine identity；
替换 Worker 不得接管旧 incarnation 的终止确认。缺失/不匹配的执行端点保留预占，不提前释放。
SSE 断连、非流式中断、建连失败使用有界、屏蔽外层 ASGI 取消作用域的清理；上游关闭与调度通知
各有 5 秒上限，关闭失败也会尝试发送 cancel。此通知只是意图，仍以 Worker 的静止与准入关闭回执
决定资源释放。超时进入服务日志，不能记为成功取消。

未配置 NATS 时，默认服务把 outbox 事件写入 INFO 级 `lifecycle_event` 日志，再确认本地事件；
关闭 INFO 不静默丢弃待发事件。配置 NATS 时使用既有可靠事件路径。日志 fallback 不是持久消息总线。
未配置 etcd 时，默认 Scheduler 使用内存 store，不得据此宣称重启对账、故障恢复率或 HA。
`tools.validate_inference_loop` 的 stdout decision ID 与这些事件对应；应核对每个 route 的
cancel/completion intent 和唯一 release，release 的 execution receipt 必须 terminal、quiescent、
admission_closed。HTTP 200 或顺序请求成功本身不是全部释放证据。

## 同机持久控制状态与重启验证

在独立测试环境使用 Compose 已固定的 etcd 镜像。先启动 etcd 作为稳定的网络 namespace 所属容器：

```bash
docker run -d --name freechat-etcd -p 127.0.0.1:8080:8080 \
  -v freechat-etcd-state:/etcd-data \
  quay.io/coreos/etcd:v3.6.5@sha256:3397341272b9e0a6f44d7e3fc7c321c6efe6cbe82ce866b9b01d0c704bfc5bf3 \
  /usr/local/bin/etcd --name=local --data-dir=/etcd-data \
  --listen-client-urls=http://127.0.0.1:2379 --advertise-client-urls=http://127.0.0.1:2379 \
  --listen-peer-urls=http://127.0.0.1:2380 --initial-advertise-peer-urls=http://127.0.0.1:2380 \
  --initial-cluster=local=http://127.0.0.1:2380
docker exec freechat-etcd /usr/local/bin/etcdctl endpoint health
```

随后使用 README 的 Worker/Gateway/Scheduler 镜像、密钥与模型参数，但三个容器均指定
`--network container:freechat-etcd`；删除 Worker 的 `-p`，端口只在 etcd namespace 所属容器发布。
Scheduler 额外传入 `-e ETCD_ENDPOINT=http://127.0.0.1:2379`。不开放 etcd 到宿主机/外网。
不要把有未决请求的内存控制面直接切换为空 etcd，这不是在线迁移流程。
停机时 etcd 最后停止，不删除 volume；此单节点启动不提供 quorum/HA。

故障验收必须有以下原始观测，全部输出到 stdout/服务日志，不创建结果目录：

1. 真实 GPU 流式请求出现首个文本 token 后，读取 etcd 中的 RequestLedger：
   记录 decision ID、Worker generation、engine instance、非 released 状态与预占字节。
2. 仅对该测试 Scheduler 注入 SIGKILL，检查容器实际退出；保持 Gateway、Worker 和 etcd 存活。
3. 继续消费流。即使 GPU 已完成，Scheduler 停机期间持久 ledger 仍不得自行释放预占。
4. 启动同一个 Scheduler。检查恢复记录的身份未变化，执行回执为 terminal/quiescent/admission_closed，
   然后才进入 released。验证新请求可准入，并等待其可信释放与 outbox 清空。
5. 合并该容器重启前后的正常日志，运行既有 `--mode audit-log`，核对所有 route/release 与实际池上限。

etcd client 必须使用写入响应的 commit revision，不能通过后续 GET 猜测本次写入结果；
CAS 竞争测试应验证只有一个胜者，删除不存在的键返回 false。
这条路径当前验证的是存活 Worker 上的 Scheduler 恢复，不代表重新执行任务、Worker 替换恢复、
etcd 故障、JetStream 重投、分区、跨节点或故障恢复成功率/RTO。恢复矩阵仍按根计划继续。

控制面镜像构建使用 BuildKit 管理的 uv 下载缓存，并保持 `uv sync --frozen`。
网络缓慢时可通过 `--build-arg UV_HTTP_TIMEOUT=120` 设置有限下载超时；不得通过改动 lockfile
或换不明依赖绕过下载失败。构建缓存不是实验结果目录，也不进入运行镜像。

## Worker 崩溃与 incarnation 边界

真实 Worker 崩溃测试与 Scheduler 重启测试必须分开。首 token 后仅终止测试 Worker，
保留 Scheduler 和 etcd；检查客户端中断而非成功结束，并记录旧 decision、generation、
engine instance 与未释放预占。随后启动同一 Worker，核对其新身份及新请求执行结果。
向新执行端点发送旧身份命令必须被拒绝，不能把新实例的 unknown 回答当成旧任务终止。

当前真实 A5000 测试确认旧请求停留在 cancel_requested，预占不会自动回收；
这是已复现的恢复缺口，不是成功恢复。轮询日志使用以下固定诊断码，不输出原始异常文本：

- `worker_not_registered`：注册表不存在该 Worker。
- `worker_generation_changed`：当前 Worker 不属于旧 generation。
- `engine_instance_changed`：generation 相同但执行引擎身份不同。
- `execution_endpoint_missing`：当前身份缺少执行端点。

这些原因都只表示无法向原执行主体取证，不能授权释放。不得删除 etcd/SQLite 状态、
改写 generation 或伪造终态回执来通过测试。后续须接入实际进程管理器的终止确认，
绑定旧 runtime 身份、关闭迟到准入，再幂等回收对应预占；任务失败/重试与容量回收分别验收。
故障测试结束可以停机，保留未决状态和正常日志，不必为了等待未实现的恢复一直运行服务。

## JetStream 中断与确认边界

持久控制状态测试环境可在同一 namespace 加入固定的 NATS 镜像：

```bash
docker run -d --name freechat-nats --network container:freechat-etcd \
  -v freechat-nats-state:/data \
  nats:2.14.6-alpine@sha256:ad7a43eb7e3337c3c38ce5d784d1461791f95f730f252d2b25eee699752a0ca3 \
  --jetstream --store_dir=/data --http_port=8222 --addr=127.0.0.1
```

Scheduler 启动时再传 `-e NATS_URL=nats://127.0.0.1:4222`，不与另一 Scheduler 并行占用同一控制端口。
首次连接及流配置合计受 10 秒 deadline 限制，失败或取消时关闭客户端，不报告 Scheduler ready。
已运行的客户端以 2 秒重连间隔持续尝试，不使用 SDK 默认 60 次耗尽即关闭的限制；
单次 JetStream 操作仍有 5 秒 timeout。服务日志以 `lifecycle_bus_connection_error` 和
`lifecycle_bus_reconnected` 表达连接状态，不记录含凭据的 URL/异常文本。
此模式事件进入 `FREECHAT_LIFECYCLE`，不能再只凭日志 fallback 判断事件已送达。
请求预占、RequestLedger.pending 和 publisher outbox 是不同状态；GPU 完成并获得可信回执后
可以释放请求容量，但发布未确认的事件仍须保留，不把消息故障误记为 GPU 执行失败。

短时中断验收：

1. 真实流式生成首 token 后，仅终止测试 NATS，保持 Scheduler、Worker 和 etcd 存活。
2. 检查 GPU 的生成结果及终态回执；记录对应 decision 的待发 event ID/type，
   要求包括 release 事件，并确认 publisher outbox 仍存在。
3. 启动同一 NATS、保留其 volume。等待两层待发记录清空，从 JetStream 读取原事件 ID，
   核对 route、completion 与 release 及 Worker 回执身份。
4. 分别测试服务端已接收但发布确认丢失，以及消费者未 ACK：前者重用原 event ID，
   后者必须观察同一 stream sequence 的重投，再 ACK 后检查 pending ack 清空。

待发事件以首次持久化的内容和时间戳为准。重试可以重建 enqueue 时间，但同一 event ID
不得更改 tenant、harness、aggregate、generation、payload 等语义字段；冲突拒绝并保留原记录。
延长中断时记录 NATS 实际停止/恢复时刻，并检查 Scheduler PID 与 StartedAt 均未改变；
恢复不仅要求连接成功，还须检查待发清空、原事件可读取、新 GPU 请求及其终态释放。
已验证 150 秒中断超过默认客户端的重连耗尽边界；该时长是故障注入持续时间，不是 RTO。
跨过去重窗口的核验允许内容一致的重投，按 event ID 归并逻辑事件，冲突内容仍必须失败。
当前 JetStream duplicate window 为 120 秒，窗口内发布去重不等于永久去重或分布式 exactly-once；
消费者仍必须处理重复。消费端持久幂等、更长/反复中断、quorum 和分区仍待独立验收。
所有探针输出到 stdout，事件保存在正常服务的 etcd/JetStream 数据卷，不建立测试结果目录。

## 开发服务停机

停止测试发流并等待已准入请求的可信释放核验后，停止 Gateway、各 Worker，最后停止 Scheduler；
共享 namespace 的 peer Worker 要先于 namespace 所属 Worker 停止。不删除 state volume 或服务日志。
每次停机命令只指定一个服务，等待其返回后再停止下一个；不要把多个名字交给同一个
`docker stop` 并假定按列表顺序停机。
Scheduler 显式处理 SIGTERM/SIGINT，清理任务、停止 gRPC 并关闭已配置的依赖；
正常日志依次出现 `scheduler_stop_requested`、`scheduler_shutdown_complete`，容器应以 0 退出。
仅有 `docker stop` 返回不代表正常退出，必须检查 `State.ExitCode`；137/强杀不能记为清理通过。
空闲容器及 NATS 不可用/重连状态的退出已验证。NATS drain 最多等待 5 秒，已关闭、重连中
或超时会记录 `lifecycle_bus_drain_unavailable`，随后关闭客户端；这不是投递确认，不能据此
删除未获发布确认的持久事件。仍不证明在途任务迁移、存储故障或其他依赖故障下的退出保证。

## 并发准入验证

使用 README 的独立同机启动路径，把 Worker 参数设置为
`--num-gpu-blocks-override 128 --max-model-len 1024`，不要与正常请求或其他 probe 混跑。
这是上游实际 KV 分配配置，不是修改注册预算来制造压力。

```bash
docker exec freechat-worker /opt/freechat/.venv/bin/python \
  -m tools.validate_inference_loop --mode concurrent --concurrent-requests 8 --rounds 3
```

每轮并发请求要求同时出现成功准入和 503 背压，成功请求必须真实生成指定 token 数；
确认释放期间允许有界重试，并统计全部拒绝。HTTP 返回不等于预占已释放，不要求零确认延迟。
随后按 stdout 的 `AUDIT_REQUIRED` 值设置下列参数，等待请求释放后核对正常 Scheduler 日志：

```bash
docker logs freechat-scheduler 2>&1 | \
  docker exec -i freechat-worker /opt/freechat/.venv/bin/python \
  -m tools.validate_inference_loop --mode audit-log \
  --worker gpu-worker --pool-bytes "$POOL_BYTES" \
  --expected-routes "$ROUTES" --expected-rejections "$REJECTIONS"
```

审计要求每个 route 恰好一次有效释放、回执身份一致、terminal/quiescent/admission_closed，
并重建峰值预占不超过实测 usable KV pool。503 只有在服务日志记录该 Worker 的
`vram_capacity` 拒绝时才计为容量背压；健康失败或网络故障不能冒充该结果。
日志不完整、事件冲突或未释放直接失败。同一事件的相同重投不重复记账。
`--expected-cancels` 用于带生成中断连的完整协议测试；这时必须匹配真实 ABORT 回执。
这些计数不是吞吐、RTO 或分布式 exactly-once 指标。所有输出仍只到 stdout 和服务日志。

## 同一 Scheduler 管理多 Worker

同机独立 Worker 可以共享第一个 Worker 的网络 namespace，并各自使用独立 GPU、HTTP/执行 RPC
端口和持久 state volume。仍然只启动一个 Gateway 与一个 Scheduler；共享控制面不等于 TP/PP、
跨节点 transport 或共享 KV tensor。两个 Worker 的 `node-id` 必须与 Gateway origin 一致，
服务 token 必须一致。以下示例接在 README 启动路径之后；做并发压力时，第一张卡也应使用
前述 128-block/1024-context 配置。

```bash
docker run -d --name freechat-peer --network container:freechat-worker \
  --gpus device=1 --shm-size 1g -e FREECHAT_WORKER_TOKEN \
  -v /absolute/path/to/Qwen2.5-0.5B-Instruct:/model:ro \
  -v freechat-peer-state:/var/lib/freechat "$WORKER_IMAGE" \
  --worker-id peer-worker --node-id local-node --model /model --served-model-name qwen \
  --host 127.0.0.1 --port 8001 --control-port 50053 \
  --scheduler-target 127.0.0.1:50051 --gpu-memory-utilization 0.25 \
  --num-gpu-blocks-override 128 --max-model-len 1024 --enforce-eager

docker exec freechat-worker /opt/freechat/.venv/bin/python \
  -m tools.validate_inference_loop --mode concurrent \
  --peer-worker-url http://127.0.0.1:8001
```

先在当前 Scheduler 的正常日志中确认两个 `worker_registered`，逐一核对 worker ID、
generation、物理 GPU 和 HTTP/执行端口；`/health` 或 `/freechat/runtime` 可用不代表注册完成。
控制面重新启动时 gRPC 重连可能延迟，禁止把只有一张卡注册的测试作为双卡结果。
若验证器报 `not every observed Worker executed the contention workload`，本轮失败，
检查注册与拒绝日志后从独立日志范围重新验收，不删失败日志或强行通过。
验证器核对不同物理 GPU UUID，根据 Gateway 的
`x-freechat-worker-id` 统计实际执行位置，要求每个声明的 Worker 都执行长请求；
不从预期路由或输入列表猜测执行位置。stdout 为每个 Worker 输出单独的 audit 参数。
对同一份 Scheduler 日志逐卡运行 `--mode audit-log --allow-other-workers`，
每张卡必须使用自己的 pool 与 route 数，不能把显存简单合并后掩盖单卡超售。
停止时先停止 peer，再停止持有共享网络 namespace 的主 Worker。
空闲 peer 正常停止后可运行主 Worker 的单卡 probe；当前 Scheduler 日志包含前后两段，
audit 需要累加同一卡的 route/rejection 数并保留 `--allow-other-workers`。
停止卡的拒绝必须明确包含 `worker_unhealthy` / `worker_draining`（或超时后的 stale 原因），
不能将它计成容量不足；正常停止不等于故障注入或在途任务恢复。
这不证明 Worker 崩溃后的在途任务迁移、跨节点部署或调度性能收益。

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
