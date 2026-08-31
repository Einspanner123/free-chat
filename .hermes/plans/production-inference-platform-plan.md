# FreeChat 生产级小模型长上下文推理平台历史计划（已冻结，待迁移归档）

> 状态：SUPERSEDED_PENDING_MIGRATION / 禁止继续按本计划新增功能
> 适用分支：`main` 及其后续开发分支
> 取代关系：本计划已由 `.hermes/plans/agent-aware-inference-infra-plan.md` 取代。历史能力只能按目标迁移矩阵选择性继承，不得继续把聊天应用、Agent Control Plane 或工作流编排作为项目主线。
> 文档职责：保留旧方向、既有决策和迁移审计线索，不再作为新增实现的上位计划。
> 删除规则：**禁止仅因代码“看起来完成”而删除本文件。只有全部阶段验收通过、证据归档完成、已知阻塞清零，并由项目负责人明确确认后，才允许单独提交删除。**

---

## 0. 文档治理与完成口径

### 0.1 本计划的权威性

- 本文件是生产化改造的上位计划和最终验收清单。
- `.hermes/plans/context-management-plan.md` 作为既有上下文设计资料保留；当它与本计划冲突时，以本计划为准。
- 每项任务必须同时满足“实现、测试、运行证据、文档”四类条件，不能以配置存在、代码可编译或单元测试通过替代端到端验收。
- 所有性能和质量结论必须注明模型、revision、引擎版本、硬件、数据集、样本量、并发、输入/输出长度和统计口径。
- 所有实验性能力默认关闭；达到质量、稳定性和回滚门槛后才能成为默认路径。

### 0.2 状态标记

- `[ ]` 未开始或未验收。
- `[~]` 已实现但证据不完整，不计入完成。
- `[x]` 已实现且本阶段验收证据齐全。
- `[!]` 阻塞，必须记录原因、负责人和解除条件。

### 0.3 删除门禁

删除本文件前必须逐项满足：

- [ ] 第 4 至第 11 节全部阶段验收项为 `[x]`。
- [ ] 第 12 节生产 SLO 在规定测试拓扑上达标。
- [ ] 第 13 节故障演练全部通过，没有未解释的数据或资源泄漏。
- [ ] 第 14 节证据包已提交，结果能够由干净环境复核。
- [ ] README、部署文档、运维手册和实际默认路径一致。
- [ ] 没有以 placeholder、mock-only 或未运行的 gated test 冒充完成的模块。
- [ ] 所有 P0/P1 问题关闭；残留 P2 问题有负责人、截止条件和不影响发布的依据。
- [ ] 项目负责人在 issue、PR 或提交说明中明确写出“同意删除生产化实施计划”。
- [ ] 删除动作使用独立提交，提交信息包含最终验收证据索引。

---

## 1. 项目目标

FreeChat 的目标不是重新实现模型训练框架、vLLM 内部调度器或向量数据库，而是整合成熟组件，提供一个统一的小模型推理控制面和聊天数据面：

1. 单机本地可用：CPU、单 GPU 或多 GPU 环境能够低门槛部署。
2. 多端可扩展：局域网和远程 GPU 节点能够注册、发现、调度和安全退出。
3. 小模型优先：通过检索、压缩、分层摘要、长上下文模型选择和缓存复用，提高 0.5B–7B 模型的有效任务能力。
4. 资源感知：调度决策考虑模型能力、GPU 显存、KV cache、排队 token、吞吐、尾延迟和缓存亲和性。
5. 生产可靠：具备准入控制、背压、租约、失败重选、熔断、优雅下线、可观测性和可审计证据。
6. 统一体验：本地模式和多节点模式复用相同 API、请求模型和调度语义，只替换基础设施适配器。

### 1.1 非目标

- 不自研 continuous batching、PagedAttention、模型并行或 GPU kernel；优先复用 vLLM/SGLang。
- 不以补齐 SFT、DPO、PPO、Reward Model 目录为主线；未完成模块不得进入生产能力清单。
- 不自研通用向量数据库、消息队列、服务发现或监控系统。
- 不把 LongBench 单项分数等同于系统生产能力。
- 不以微服务数量、测试函数数量或 README 篇幅作为完成指标。

### 1.2 设计原则

- **复用优先**：选择成熟执行引擎和基础设施，自研价值集中在跨节点策略和系统闭环。
- **能力先过滤，代价后排序**：先排除不支持请求的 worker，再对候选实例评分。
- **测量代替猜测**：资源状态来自 worker 心跳和引擎指标，不以 Redis 人工计数作为唯一真相。
- **租约代替裸计数**：所有资源预留必须绑定 request ID、TTL 和幂等释放。
- **退化可解释**：上下文压缩、模型降级、远端转发和拒绝请求都必须返回原因。
- **默认安全**：多租户缓存隔离、鉴权、传输安全和隐私边界必须先于跨设备共享。
- **失败是正常路径**：设计必须覆盖节点崩溃、网络分区、OOM、超时、取消和控制面短暂不可用。

### 1.3 已冻结的最终产品定义

以下决定来自项目负责人的逐项 grilling 确认。除非负责人明确发起架构变更并记录 ADR，否则实施过程中不得以开发时长、人力估计或局部实现便利为由缩减、替换或绕过。

FreeChat 的第一身份是：

> 面向面试官和开源项目读者的分布式小模型推理控制面。系统将多台异构 NVIDIA GPU 组织为独立推理 Worker，同时接入 OpenAI-compatible 高阶模型；本地模型处理简单任务、记忆提取和受隐私约束的执行，高阶模型负责复杂任务规划，最终由本地 Policy Engine、Temporal Workflow 和资源感知 Scheduler 完成安全执行。

产品不是跨节点拆分同一个模型，而是分布式多模型协作系统：

- 每个 Worker 是独立模型实例；请求或 DAG Step 在 Worker 之间调度。
- 不做跨局域网 tensor parallel、pipeline parallel 或 collective inference。
- 不做跨 Worker KV cache 迁移或持久化恢复。
- Worker 内 prefix cache 复用底层引擎原生能力；控制面维护 cache affinity 元数据并进行 cache-aware routing。

### 1.4 已冻结的真实验收拓扑

物理环境和首个验收拓扑固定为：

```text
ross
  Ubuntu 24.04
  Control Plane + RTX A6000 48GB Worker
  125GiB RAM
  模型/实验数据使用 /media/ross/8TB

workstation
  Ubuntu 22.04
  RTX A5000 24GB Worker
  RTX A4000 16GB Worker
  模型/实验数据使用 /data

network
  Tailscale overlay
  当前允许 DERP 路径作为正式验收拓扑
  Scheduler 必须感知 RTT、抖动、丢包、带宽和历史流式停顿
```

首版只承诺 Linux + NVIDIA。三张 GPU 作为三个独立 Worker：

- A6000：复杂本地任务、较大模型和长上下文高 KV 容量。
- A5000：常规执行和中等规模模型。
- A4000：Router、分类、事实提取和简单任务。
- 具体模型 ID、revision 和量化必须通过 Model Registry 配置并在基线阶段验证；不在业务代码中硬编码。

### 1.5 已冻结的部署与基础设施职责

- Docker Compose 是本地开发和最小部署路径。
- kubeadm Kubernetes 是生产验收路径；交付标准 Helm chart，不绑定非标准发行版接口。
- 当前真实拓扑允许单控制面恢复验证；架构必须支持未来扩展至三控制节点 HA，不得把双节点伪装成 quorum HA。
- Kubernetes 负责 Pod、Service、Endpoint、readiness、网络和生命周期。
- FreeChat Worker Registry 负责模型、GPU、KV、队列、性能、网络路径和隐私等级。
- Consul 仅是当前原型资产，迁移完成后从生产主路径和失效配置中删除。
- Temporal 负责 DAG durable execution、重试、定时器、Human Approval 等待和崩溃恢复。
- RocketMQ 不再承担核心职责；Temporal 迁移完成后删除 RocketMQ 生产依赖和死代码。
- PostgreSQL 是消息、MemoryFact、Conflict、Audit和业务元数据的权威事实源。
- pgvector 是可重建的语义检索索引；不引入独立向量数据库。
- Redis 只用于 admission token bucket、短期 reservation/TTL、幂等键、短期路由提示和 ephemeral session 状态，不保存权威业务事实。

### 1.6 已冻结的 Planner、Workflow 与调度权限

自动路由采用分层级联：

```text
确定性规则
  → 本地轻量 Router
  → 复杂或不确定任务调用高阶云端 Planner
  → 本地 Plan Validator
  → 本地 Policy Engine
  → Temporal
  → Scheduler
  → Worker
```

- Planner 输出 JSON Schema 约束的结构化 DAG，不输出自然语言后由执行器猜测。
- 第一版 DAG 节点限定为 LLM inference、Memory retrieval、Human approval、Retry/fallback。
- Plan 不允许原地篡改；变化生成新 revision、parent revision、差异和原因。
- Planner 只能声明硬能力需求和软 placement preference，不得看到或指定具体 Worker ID。
- Scheduler 对 placement 拥有最终选择权和否决权。
- Planner 输出无效时只允许同一 Planner 自动修复一次；仍无效则明确失败，不把非法 Plan 交给执行器“尽量运行”。
- 普通用户可以硬性指定模型、`local-only` 和是否允许云端 fallback；只有 Operator/Admin 在调试、benchmark 和故障复现时能够 Pin 具体 Worker。
- 用户可查看自动路由原因，并选择本地高质量或云端高质量重新运行。
- 提供 `private`、`economy`、`balanced`、`quality` Auto Profile；每次实际决策仍需记录原因。

Human Approval 规则固定为：

- 永不因超时自动批准或拒绝。
- 进入等待状态时释放 GPU reservation、KV、网络连接和其他短期资源。
- Workflow 持久化为 `WAITING_APPROVAL`，返回可恢复 response/workflow ID。
- 恢复前重新验证 Plan revision、模型、权限、凭据、记忆、隐私、预算和节点状态。
- 环境变化时必须向用户说明，不得直接沿用旧决定。
- 支持 UI、站内状态、Webhook 和邮件 Notification Provider。

### 1.7 已冻结的长期记忆与冲突语义

长期记忆第一验收目标：

1. 至少 1000 轮对话后召回早期明确事实并给出来源。
2. 支持长期项目状态、决策、依赖、TODO 和变更历史。
3. 来源必须包含不可变 message ID、session ID、session turn、全局时间和原始时间戳。

记忆形态采用组合方案：

- 近期原文；
- 带来源的结构化 MemoryFact；
- 分层/阶段摘要；
- pgvector 语义检索。

记忆提取在回答完成后异步执行：

- 用户明确陈述且低风险的事实可以进入 `ACTIVE`。
- 推断事实进入 `PENDING_REVIEW`。
- 高风险约束进入 `PENDING_CONFIRMATION`。
- 与既有事实冲突时必须进入专门的 `CONFLICT_PENDING`，永远不能由模型自动脱离冲突状态。

冲突解决必须：

- 每次相关请求重新询问，除非用户已经给出准确解决方案。
- 展示旧事实、新事实、日期、会话、轮次和消息来源。
- 提供保留旧值、接受新值、拆分/合并作用域、自定义准确约束和暂不处理。
- 自然语言回复只能生成 `ResolutionProposal`；本地 Validator 确认无歧义后才提交状态转换。
- 冲突、每次询问、用户回复、解析提案和最终解决全部写入不可变事件记录。
- 根据风险决定阻塞范围，但不改变“必须用户裁决”：普通信息可带提示继续；项目约束阻塞相关 Step；安全/隐私和破坏性冲突阻塞所有可能违规动作。

记忆作用域和优先级固定为：

```text
当前显式用户指令
  > PROJECT
  > USER
  > SESSION
  > GLOBAL_SYSTEM
```

优先级只用于检索与提示，不得静默覆盖冲突。

删除语义固定为：

- 普通删除是可恢复软删除。
- 用户可发起不可恢复 purge；必须级联 Fact、embedding、summary、index 和 cache。
- 只保留不含被删除内容的审计事件。
- 删除原始消息时询问用户是否级联删除派生记忆，默认建议级联。

### 1.8 已冻结的 API、Provider 与用户界面

API 采用双层表面：

- `/v1/responses` 是 FreeChat 内部主接口，表达 Plan、Step、Approval、Memory provenance 和 Trace 事件。
- `/v1/chat/completions` 是兼容适配层，不改变标准字段语义。
- Chat Completions 遇到审批时发送 `freechat` 命名空间扩展事件，结束长连接并返回 resume ID，不能无限保持 HTTP 连接。
- Planner 使用支持 JSON Schema Structured Outputs 的 Provider 能力；Provider 不支持时必须明确降级或拒绝，不能假装严格结构化。

模型命名固定为：

```text
auto
local/auto
local/<model>
planner/auto
remote/<provider>/<model>
```

- `auto` 允许本地和远端，但必须通过隐私和预算策略。
- `local/auto` 禁止云端。
- `planner/auto` 只请求规划。
- 远端通过通用 OpenAI-compatible Provider 接口；真实验收至少覆盖一个 OpenAI 官方 Provider 和一个第三方 compatible endpoint。
- 同时支持管理员共享凭据和用户私有凭据。
- External Secrets Operator 对接可插拔 Secret Store；FreeChat 只保存 credential handle 和授权元数据。

Web UI 保持轻量但必须完整展示系统能力，固定五页：

1. Chat：聊天、Auto Profile、local-only、模型/fallback约束和路由解释。
2. Execution：Plan revision、DAG、Step状态、审批和恢复。
3. Workers：模型、GPU、显存、KV、队列、网络路径、健康和drain。
4. Memory：事实、来源、冲突、修改、软删除和purge。
5. Trace：从聊天请求跳转到完整跨服务Trace和调度分解。

前端边界：React + TypeScript、轻量组件库、浏览器桌面适配；所有能力必须有 API；不开发原生移动端、Electron或依赖 UI 的私有后门。

### 1.9 已冻结的隐私、安全、发布与数据治理

- 节点信任由 mTLS 身份和可配置 `trusted_private/trusted_remote/untrusted_remote/cloud_provider` 等级共同决定。
- Tailscale 节点不因位于 overlay 网络就自动获得数据访问权。
- 云端上下文根据隐私等级使用相关片段或本地脱敏摘要；敏感会话禁止云端。
- 云端预算按用户、Provider、模型和单 Plan 限制 token、金额和调用次数。
- 额度不足时，有安全本地 fallback 才能降级；否则请求用户批准额外成本。
- 角色固定为 User、Operator、Admin、Auditor；Operator默认看不到用户正文。
- 管理员读取用户记忆只能通过 break-glass，必须填写原因并生成不可删除审计事件。
- 可观测性固定为 OpenTelemetry Collector + Prometheus + Tempo + Loki + Grafana；敏感 prompt、memory正文和API key不得进入Telemetry。
- 默认不向项目维护者上传任何遥测，只有部署者主动开启才允许外发。
- 项目采用 Apache-2.0 许可证。
- 发布带版本和 digest 的预构建镜像并保留 Dockerfile；生产禁止依赖 `latest`。
- 模型权重不进入镜像；通过宿主机缓存/PV挂载和受控下载工具管理，并校验revision和完整性。
- 完全本地模式在首次取得模型后可离线运行聊天、记忆、调度、管理和基础本地规划；云端能力不可用时必须显示降级。
- 备份覆盖 PostgreSQL、配置、加密凭据metadata、Temporal状态和必要证据文件；embedding、Prefix KV和Redis短期状态可重建，不作为权威备份。
- 管理员和用户可配置数据保留周期；不同数据类型分别过期。
- 数据导出同时提供机器可读 JSON 和人类可读报告，覆盖对话、记忆、provenance、冲突和Plan历史。

### 1.10 黄金演示与不可删减验收域

最终黄金演示固定为一条端到端故事：

```text
Chat balanced 请求
  → 本地 Router 判定复杂
  → 云端 Planner 输出结构化 DAG
  → 本地 Policy 检查隐私和预算
  → Temporal 持久化 Workflow
  → Memory 找到早期 MySQL 项目约束
  → 新 PostgreSQL 要求触发 CONFLICT_PENDING
  → 相关 Step 等待 Human Approval
  → 用户将作用域拆分为旧服务 MySQL、新项目 PostgreSQL
  → 冲突及解决事件持久化并可溯源
  → Scheduler 比较三张 GPU 的能力、KV、队列、DERP网络和cache affinity
  → 选择预计端到端 TTFT 最优 Worker
  → vLLM 流式执行
  → UI 展示 Chat、Execution、Workers、Memory 和 Trace
  → OpenAI-compatible API 重放能力
  → 模拟 Worker 故障，验证取消、reservation 回收和 fallback
```

最终验收域不可删减：

1. 多节点资源调度与故障恢复。
2. Planner DAG、持久审批、revision和Policy约束。
3. 1000轮长期记忆、provenance、冲突、删除和恢复。
4. Responses/Chat Completions API与五页UI。
5. 可观测性、权限、凭据、mTLS、备份和数据治理。

---

## 2. 当前基线与必须纠正的问题

以下是开始实施时的已知基线，完成过程中必须重新核对：

### 2.1 已有可复用资产

- Go API Gateway、Auth、Chat Service 和 gRPC 流式链路。
- Python HF/vLLM 推理后端；vLLM 已提供 continuous batching、PagedAttention、TP 和原生 prefix caching。
- Consul 健康实例发现。
- Redis Lua 最小活跃请求数选择原型。
- 控制面/计算面 Compose 初步拆分。
- `context-engine` 的路由、检索、压缩、布局与 Go 远程回退接口。
- 单元测试、gated GPU E2E 和 Go↔Python 跨进程测试框架。

### 2.2 P0 基线缺陷

- [ ] 推理服务必须真正使用显式 `ADVERTISE_HOST/PORT`；禁止在容器中通过访问 `8.8.8.8` 猜测多机可达地址。
- [ ] `context-engine` 必须进入部署拓扑；解决与推理服务的 `8089` 端口/文档冲突。
- [ ] worker 注册必须描述模型和硬件能力，不能只注册统一的 `llm-inference` 名称。
- [ ] 调度状态必须从裸 ZSET 加减升级为带 request ID 和 TTL 的 reservation。
- [ ] 负载释放必须幂等且有下限，进程崩溃后能够自动校准。
- [ ] 所有被称为“主路径”的服务必须能由受支持的部署命令启动并通过 readiness。
- [ ] 固定 `container_name`、固定单实例宿主机端口和 `GPU count: all` 不得阻止单节点多 worker。
- [ ] 结果 JSON 必须带 commit、环境、模型 revision、数据版本和运行命令。

---

## 3. 目标架构与核心契约

### 3.1 逻辑架构

```text
Client / Web / OpenAI-compatible API
                │
                ▼
        API Gateway / Auth
                │
                ▼
       Inference Control Plane
       ├─ Model Registry
       ├─ Worker Registry
       ├─ Admission Controller
       ├─ Context Cost Estimator
       ├─ Scheduler
       ├─ Reservation/Lease Manager
       ├─ Retry/Circuit Breaker
       └─ Observability/Audit
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
 Local Worker  LAN Worker  Remote Worker
 HF/vLLM       vLLM       vLLM/SGLang
 CPU/GPU       1..N GPU   TP/Quantized
       └──── heartbeat + capacity + metrics ────┘
```

### 3.2 Worker 能力模型

必须以版本化 schema 表达，至少包含：

```text
WorkerDescriptor
  schema_version
  worker_id / node_id / boot_id
  advertise_endpoint
  engine / engine_version
  model_id / model_revision / tokenizer_revision
  dtype / quantization
  gpu_type / gpu_count / gpu_ids
  tensor_parallel_size
  total_vram_bytes / allocatable_vram_bytes
  max_model_len / max_batch_tokens
  supported_features
  tenant_scope / locality
  started_at / draining / health_epoch
```

### 3.3 Worker 动态状态

```text
WorkerStatus
  observed_at / sequence
  free_vram_bytes
  kv_cache_used_bytes / kv_cache_free_blocks
  running_requests / waiting_requests
  running_sequences / waiting_sequences
  pending_prefill_tokens / pending_decode_tokens
  recent_tps / ttft_p95 / tpot_p95
  prefix_cache_entries / prefix_cache_hit_rate
  oom_count / error_rate
  ready / degraded / draining
```

动态状态必须有过期时间；过期 worker 不得进入候选集。

### 3.4 请求成本模型

```text
InferenceRequestSpec
  request_id / tenant_id / session_id
  requested_model / acceptable_fallbacks
  prompt_tokens / max_new_tokens
  context_policy / privacy_scope
  prefix_hash / adapter_id
  priority / deadline / latency_slo
  locality_preference
  required_features
```

请求进入调度器前必须完成 token 估计和上限校验。估计失败时使用保守值，不允许按零成本进入队列。

### 3.5 Reservation 契约

```text
Reservation
  reservation_id / request_id
  worker_id / model_revision
  estimated_prefill_tokens / estimated_decode_tokens
  estimated_kv_bytes
  state: pending | admitted | running | completed | cancelled | expired
  created_at / expires_at / version
```

- 创建、续租、状态迁移和释放必须幂等。
- worker 未确认前为 `pending`；确认后才能进入 `admitted/running`。
- 超时或进程崩溃后由 TTL 和 reconciliation 自动回收。
- 请求重试必须生成新 reservation，并保留父请求关联，避免双重计费。

### 3.6 调度流程

```text
validate request
  → resolve model policy
  → estimate context/token/KV cost
  → capability filter
  → health/freshness filter
  → admission check
  → score candidates
  → create reservation lease
  → dispatch
  → confirm running
  → stream/cancel/complete
  → idempotent release
  → reconcile observed state
```

初始评分函数应可解释、可配置，并记录每个分量：

\[
score(w,r)=
\alpha Q_w+
\beta \frac{P_r}{T^{prefill}_w}+
\gamma \frac{D_r}{T^{decode}_w}+
\delta M_{kv}(r,w)-
\eta H_{prefix}(r,w)+
\lambda R_{risk}(w)+
\mu C_{network}(w)
\]

禁止在没有消融和线上/压测证据时宣称该公式“最优”。

---

## 4. Phase 0：基线冻结与可复现实验框架

### 4.1 任务

- [ ] 建立 `docs/architecture/current-state.md`，以代码为准记录当前调用链、端口、配置来源和已知断点。
- [ ] 建立 `docs/architecture/target-state.md`，记录本计划目标架构及关键 ADR。
- [ ] 为所有 Compose 文件运行配置展开校验，消除未定义变量、端口冲突和文档差异。
- [ ] 建立轻量 evidence manifest schema，包含 commit、dirty 状态、命令、环境、硬件、模型和数据 revision。
- [ ] 固化三套基线场景：本地 CPU/HF、单 GPU/vLLM、双节点远程 vLLM。
- [ ] 记录当前 TTFT、TPOT、吞吐、失败率、GPU/KV 占用和调度结果，禁止只记录平均值。

### 4.2 验收

- [ ] 干净 checkout 能生成同结构的 evidence manifest。
- [ ] 所有 README 端口与实际配置一致。
- [ ] 基线结果可追溯到唯一 commit 和运行命令。
- [ ] 当前已知缺陷以 issue/清单形式登记，没有“代码里以后再说”的隐含问题。

---

## 5. Phase 1：部署与 Worker 注册正确性

### 5.1 配置统一

- [ ] 定义单一配置优先级：CLI > environment > config file > default。
- [ ] 统一 `APP_ENV/ENVIRONMENT`、`APP_PORT/GRPC_PORT`、模型和引擎变量命名。
- [ ] 启动时输出脱敏后的 resolved config，并将配置摘要写入实例元数据。
- [ ] 未知或冲突配置必须 fail fast，不允许静默使用错误默认值。

### 5.2 Worker 注册

- [ ] 使用显式 `ADVERTISE_HOST/ADVERTISE_PORT`；本地模式才允许受控自动探测。
- [ ] 注册 `worker_id/node_id/boot_id`，重启后能够区分旧实例。
- [ ] 注册模型 revision、tokenizer revision、量化、窗口和 GPU 能力。
- [ ] readiness 只有在模型加载、warmup 和健康探测完成后才能为 true。
- [ ] 支持 `draining`：停止接收新请求，等待在途请求完成后注销。
- [ ] 心跳带单调 sequence，拒绝旧状态覆盖新状态。

### 5.3 部署 profile

- [ ] `local-cpu`：最少依赖、无 GPU、可选内嵌/轻量状态后端。
- [ ] `local-gpu`：单机一个或多个 worker，可显式绑定 GPU。
- [ ] `multi-node`：独立控制面和多个计算节点。
- [ ] Docker Compose 作为开发/最小部署路径；配置与生产领域契约一致。
- [ ] 使用 kubeadm 建立生产验收集群：`ross` control-plane + worker，`workstation` worker。
- [ ] 提供发行版无关的 Helm chart；不得依赖仅 k3s 可用的私有能力。
- [ ] 当前单控制面具备备份/恢复和重建演练；文档说明三控制节点 HA 扩展方法，不宣称双节点 quorum HA。
- [ ] Kubernetes 负责 Service/Endpoint/readiness；FreeChat Worker Registry 负责模型和资源能力；生产路径不再依赖 Consul。
- [ ] 移除阻止扩缩的固定容器名和固定实例端口假设。
- [ ] `context-engine` 进入本地和多节点部署，并拥有独立端口和 health/readiness。
- [ ] 模型缓存目录可挂载、可复用、可校验 revision，避免每次构建镜像下载权重。
- [ ] `ross` 模型/PV使用 `/media/ross/8TB`，`workstation` 使用 `/data`；根分区不得承载大模型权重。
- [ ] 发布带版本和 digest 的预构建镜像并保留 Dockerfile；生产配置禁止 `latest`。
- [ ] 模型不打进镜像；受控下载工具校验 model/tokenizer revision 和文件完整性。

### 5.4 验收

- [ ] 同一节点可启动两个不同模型或同模型不同 GPU worker。
- [ ] 两台机器间 Kubernetes/Worker Registry 返回的 endpoint 从调用方真实可达。
- [ ] worker 启动、重启、drain、退出的注册状态无幽灵实例。
- [ ] 模型未加载完成时不会接收流量。

---

## 6. Phase 2：资源遥测与模型注册表

### 6.1 遥测采集

- [ ] 从 vLLM/SGLang 原生指标获取运行/等待序列、KV block 和吞吐；不重复实现引擎内部统计。
- [ ] GPU 指标优先接入 DCGM exporter 或 NVML 适配器。
- [ ] 上报 free/used VRAM、GPU utilization、温度、ECC/Xid 错误等必要健康信号。
- [ ] 上报 per-model TTFT、TPOT、tokens/s、错误率和 OOM。
- [ ] 指标包含 worker、node、model revision、engine 和 quantization 标签，同时控制标签基数。
- [ ] 网络主动探测采集 RTT、抖动、丢包、可用带宽和 path type（LAN/Tailscale direct/DERP/public）。
- [ ] 使用真实请求持续校正 stream stall、网络吞吐和端到端延迟预测。
- [ ] Worker trust level 与网络 path 分离；Tailscale endpoint 不自动获得可信权限。

### 6.2 模型注册表

- [ ] 定义模型规范：模型 ID、revision、tokenizer、context window、chat template、资源需求和允许后端。
- [ ] 区分逻辑模型名和具体 deployment revision。
- [ ] 支持 fallback policy，但默认不得跨隐私域或自动更换语义不兼容模型。
- [ ] 启动时校验 worker 实际模型与声明一致。
- [ ] 记录模型加载失败、校验和失败和不兼容原因。

### 6.3 状态存储

- [ ] 静态 descriptor 和动态 status 分离。
- [ ] 动态状态有 TTL，scheduler 对超时状态 fail closed。
- [ ] Redis、Worker Registry或Kubernetes API短暂不可用时定义清晰行为：继续在途请求、停止新 admission、恢复后 reconcile。
- [ ] 提供状态版本迁移和向后兼容测试。

### 6.4 验收

- [ ] Dashboard 能展示每个 worker 的模型、GPU、KV、队列和延迟。
- [ ] 停止心跳后，worker 在规定时间内退出候选集。
- [ ] 指标与 vLLM 原始观测误差在约定范围内。
- [ ] scheduler 不再仅依赖 Redis 活跃请求计数。

---

## 7. Phase 3：准入、Reservation 与资源感知调度

### 7.1 请求预检

- [ ] 使用目标 tokenizer 精确计数；无法加载时采用保守估算并记录误差标记。
- [ ] 校验 prompt、max output、总上下文和消息结构上限。
- [ ] 估算 KV 成本，至少考虑层数、KV heads、head dim、dtype、序列长度和并发序列数。
- [ ] 请求成本纳入 tenant、priority、deadline 和 locality。

### 7.2 Admission Control

- [ ] 从 IP QPS 升级为 tenant + token-cost 限流。
- [ ] 区分 prefill token、decode token 和 KV reservation。
- [ ] 定义接受、排队、降级和拒绝策略。
- [ ] 返回结构化 overload reason 和 `Retry-After`。
- [ ] 限制 per-tenant/global waiting tokens，所有内部队列必须有界。
- [ ] Redis 限流故障时，高成本推理不得无条件 fail-open。

### 7.3 Reservation

- [ ] 使用 request ID 创建带 TTL 的幂等 reservation。
- [ ] 实现 create/confirm/renew/complete/cancel/expire 状态机。
- [ ] worker 和 control plane 重启后能够 reconciliation。
- [ ] 防止重复释放、负数计数和泄漏。
- [ ] 记录估算成本与实际成本的偏差，用于校正模型。

### 7.4 Scheduler v1

- [ ] capability filter：模型、revision、窗口、量化、adapter、引擎特性。
- [ ] health filter：ready、fresh、非 draining、错误率和 OOM 风险。
- [ ] capacity filter：显存、KV block、waiting token 和并发上限。
- [ ] score：队列、prefill/decode 预测、缓存亲和、网络和风险。
- [ ] 调度日志记录候选集、淘汰原因、各分量和最终选择。
- [ ] 提供 round-robin、least-request、token-aware、cache-aware 对照策略。

### 7.5 验收

- [ ] 大请求不会因“活跃请求数相同”被错误发送到容量不足节点。
- [ ] admission 在压力超过上限时稳定拒绝或排队，不发生无界增长。
- [ ] crash 后 reservation 自动回收，观测状态最终一致。
- [ ] 调度策略可通过回放测试确定性复现。
- [ ] token-aware 相对 least-request 在预设异构负载上改善 SLO，若没有改善则不得默认启用。

---

## 8. Phase 4：故障恢复、取消与生命周期

### 8.1 超时与取消

- [ ] 定义 connect timeout、queue timeout、TTFT timeout、idle stream timeout 和 total deadline。
- [ ] 客户端取消必须传递至 chat service、worker 和 vLLM request abort。
- [ ] 取消后 reservation、KV/sequence 和连接状态在限定时间内释放。
- [ ] HF 后端若无法安全取消，必须限制并发并明确降级语义。
- [ ] Temporal Workflow 进入 `WAITING_APPROVAL` 时释放 GPU/KV/reservation/连接，仅持久化可恢复状态。
- [ ] Approval 永不自动批准或拒绝；恢复前重新验证 Plan、权限、凭据、预算、记忆和 Worker 状态。
- [ ] Chat Completions 审批事件关闭流并返回 resume ID；Responses API 支持持久恢复。

### 8.2 Retry 与 Failover

- [ ] 仅对可安全重试且尚未向客户端输出 token 的错误进行自动重试。
- [ ] 输出开始后禁止静默换实例造成重复或不连续文本。
- [ ] retry 使用新 reservation，并保留 attempt 链。
- [ ] 实现 per-worker/per-model circuit breaker 和冷却窗口。
- [ ] 模型 OOM、不可达、deadline 和业务错误使用不同策略。
- [ ] Planner JSON Schema 校验失败只自动修复一次；仍失败则终止，不执行非法 Plan。
- [ ] Plan 变化生成不可变 revision 链，不原地覆盖已审计 Plan。

### 8.3 优雅升级

- [ ] worker drain 后不接收新请求。
- [ ] 模型切换和 revision 升级支持滚动发布和快速回滚。
- [ ] control plane 升级不丢失在途 reservation。
- [ ] schema 变更包含兼容窗口和迁移测试。

### 8.4 验收

- [ ] kill worker、断网、OOM、Redis 短暂不可用等演练结果符合策略。
- [ ] 客户端取消后无持续 GPU 计算和资源泄漏。
- [ ] 重试不会产生重复落库回复或重复计费。
- [ ] circuit breaker 能隔离故障实例并自动探测恢复。

---

## 9. Phase 5：超长上下文与缓存感知路由

### 9.1 上下文数据模型

- [ ] 移除固定“最近 10 条消息”作为唯一上下文来源。
- [ ] 消息持久化精确 token count、模型 tokenizer revision 和摘要血缘。
- [ ] 区分近期原文、阶段摘要、长期事实、文档片段和用户显式记忆。
- [ ] 摘要必须引用原始消息范围，支持失效、重算和审计。
- [ ] 处理编辑、删除、租户隔离和隐私清除。
- [ ] 定义 `ACTIVE/PENDING_REVIEW/PENDING_CONFIRMATION/CONFLICT_PENDING` 及解决终态。
- [ ] Conflict 使用 append-only event 记录每次询问、回复、ResolutionProposal、校验和最终状态转换。
- [ ] Provenance 同时保存 message ID、session ID、session turn 和 timestamp。
- [ ] 普通删除可恢复；purge 级联 Fact、embedding、summary、index 和 cache，只保留不含正文的审计事件。
- [ ] 删除原始消息时询问是否级联派生记忆，默认建议级联。

### 9.2 Context Planner

- [ ] 根据请求意图、模型窗口、token 预算和资源成本选择上下文策略。
- [ ] 同预算下比较 truncation、retrieval、summary、hybrid，而非只与弱截断下界比较。
- [ ] narrative、counting、global synthesis 等失败边界必须进入路由策略。
- [ ] 资源不足时允许：压缩、选更长窗口模型、路由到高显存节点或拒绝。
- [ ] 每次决策记录原始 token、保留 token、策略、来源和质量风险。

### 9.3 Prefix/KV Cache 亲和性

- [ ] 计算包含 model revision、tokenizer、chat template、tenant scope 的 prefix hash。
- [ ] scheduler 将安全可复用的 prefix cache 命中作为负分奖励。
- [ ] 多租户默认禁止共享敏感前缀；共享策略必须显式配置。
- [ ] cache 元数据有容量、TTL、版本和逐出策略。
- [ ] 命中率、节省 prefill token、显存成本和隐私风险均可观测。
- [ ] HF 自研 prefix path 保持实验/兼容后端；生产默认优先使用 vLLM 原生实现。

### 9.4 质量验收

- [ ] 建立长对话事实召回、长文档 QA、定位、综合生成和计数任务集。
- [ ] 报告绝对指标、置信区间、样本量和失败案例。
- [ ] 压缩质量与系统成本共同评价：质量、TTFT、GPU memory、tokens/s。
- [ ] 任何自动降级不得在未记录的情况下改变用户语义要求。
- [ ] 至少 1000 轮合成长期对话验证早期事实召回和 provenance 准确性。
- [ ] 包含 MySQL/PostgreSQL 项目约束冲突、长期未处理冲突和自然语言拆分作用域案例。
- [ ] 记忆评测报告 extraction precision/recall、冲突检测率、误报率、provenance准确率和解决状态正确率。

---

## 10. Phase 6：可观测性、安全与运维

### 10.1 可观测性

- [ ] 接入 OpenTelemetry，trace ID 贯穿 gateway、chat、scheduler、context 和 worker。
- [ ] Prometheus 指标覆盖请求、调度、队列、模型、GPU、KV、缓存和错误。
- [ ] OpenTelemetry Collector 输出 Trace 到 Tempo、日志到 Loki，Grafana提供统一入口。
- [ ] 结构化日志包含 request/tenant/session/worker/model/attempt，但不得记录敏感 prompt 原文。
- [ ] 提供本地轻量 dashboard 和多节点生产 dashboard。
- [ ] 为 TTFT、错误率、OOM、心跳过期、队列和 reservation 泄漏建立告警。
- [ ] 默认不向项目维护者上传任何遥测；外部遥测必须由部署者显式启用。

### 10.2 安全

- [ ] 外部 API 鉴权和租户身份不能只依赖客户端传入 user ID。
- [ ] 节点间使用 TLS/mTLS 或受控私有网络；禁止默认明文暴露控制端口。
- [ ] 管理 API 与推理 API 分权。
- [ ] secrets 不进入仓库、镜像层或日志；提供轮换方法。
- [ ] prefix/context cache 具有租户命名空间和删除能力。
- [ ] 防止 SSRF、任意模型下载、恶意 chat template 和不受信任 remote code。
- [ ] 定义本地隐私模式：请求和上下文不得离开指定节点。
- [ ] 角色实现 User、Operator、Admin、Auditor；Operator默认不能读取用户正文。
- [ ] break-glass读取用户记忆必须填写原因，并生成不可删除审计事件。
- [ ] External Secrets Operator 对接可插拔 Secret Store；同时支持管理员共享凭据和用户私有凭据。
- [ ] 云端预算按用户/Provider/模型/Plan限制 token、金额和调用次数；额外成本需要明确Approval。

### 10.3 运维

- [ ] 提供安装、升级、回滚、备份、恢复、drain 和故障排查手册。
- [ ] readiness/liveness/health 各自语义清晰。
- [ ] 控制面单点风险被记录；生产 profile 提供 HA 或明确非 HA 边界。
- [ ] 模型下载、校验、warmup 和磁盘清理有可观测状态。
- [ ] 备份覆盖 PostgreSQL、配置、credential metadata、Temporal状态和必要证据；定期执行恢复演练。
- [ ] 支持按全局、用户、项目和会话配置数据保留策略。
- [ ] 提供 JSON 和人类可读的数据导出，覆盖对话、记忆、provenance、冲突和Plan历史。
- [ ] 首次获取模型后，完全本地模式可在断网环境运行聊天、记忆、调度、管理和基础本地规划。

---

## 11. Phase 7：测试、压测与发布

### 11.1 自动化测试矩阵

| 层级 | 必须覆盖 |
|---|---|
| Unit | schema、成本估计、过滤、评分、reservation 状态机、幂等释放 |
| Property/Fuzz | 状态迁移、重复事件、乱序心跳、负载不为负、评分确定性 |
| Contract | Go/Python proto、版本兼容、错误码、metadata |
| Integration | Kubernetes/Worker Registry/Redis/Temporal/Chat/Scheduler/Context/Worker 的真实进程链路 |
| GPU E2E | vLLM/HF、流式、取消、prefix hit、OOM/fallback |
| Multi-node | 可达地址、注册、调度、断网、drain、升级 |
| Load | 稳态、突发、长短请求混合、异构 GPU、多租户公平性 |
| Soak | 至少覆盖长期 reservation、连接、GPU/KV 和内存泄漏 |
| Security | 鉴权绕过、租户隔离、缓存泄漏、非法配置和恶意输入 |

### 11.1.1 API 与 UI 验收

- [ ] `/v1/responses` 完整表达 Plan、Step、Approval、Memory provenance 和 Trace事件。
- [ ] `/v1/chat/completions` 通过兼容测试，扩展字段只位于 `freechat` 命名空间。
- [ ] 真实验证一个 OpenAI 官方 Provider 和一个第三方 OpenAI-compatible endpoint。
- [ ] Web UI 实现 Chat、Execution、Workers、Memory、Trace 五页，所有核心操作同时有API。
- [ ] 用户能查看路由原因并选择本地高质量/云端高质量重试。
- [ ] 实现 `private/economy/balanced/quality` Profile、`local-only` 和 fallback 控制。

### 11.2 测试真实性要求

- mock 测试只能证明局部契约，不能替代真实进程/GPU/多节点验证。
- gated test 必须在发布流水线或证据流程中实际运行，不能仅因文件存在而记为通过。
- 压测输入必须包含短 prompt、超长 prompt、短输出、长输出和混合到达分布。
- 每个性能结论至少包含 warmup、重复次数、分位数和原始结果摘要。

### 11.3 发布策略

- [ ] feature flag 控制新 scheduler、context planner 和 cache-aware routing。
- [ ] shadow mode 记录新旧调度决策差异，不实际切流。
- [ ] 小比例 canary 后逐步扩大。
- [ ] 明确自动回滚阈值。
- [ ] 保留 least-request 基线作为短期紧急回退，但修复其租约一致性。
- [ ] 模型 revision 通过新实例启动、warmup、canary、drain旧实例发布，禁止原地替换。
- [ ] 一个 Worker 实例只加载一个 model revision；同一GPU多Worker必须通过资源准入防止显存超配。
- [ ] 发布前运行黄金演示，并保存完整 Trace、Workflow、Memory Conflict 和故障恢复证据。

---

## 12. 生产 SLO 与量化验收

最终阈值必须结合目标硬件重新校准；在校准前，以下是最低验收框架而非宣传数字。

### 12.1 正确性和可用性

- [ ] 调度到不兼容模型/窗口/量化实例的次数为 0。
- [ ] reservation 重复释放、负数负载和永久泄漏为 0。
- [ ] 健康状态过期后，worker 在约定窗口内退出候选集。
- [ ] 单 worker 故障时，尚未输出 token 的可重试请求达到约定恢复率。
- [ ] 客户端取消后，GPU 序列和 reservation 在约定时间内清理。

### 12.2 延迟与吞吐

- [ ] 报告 TTFT p50/p95/p99。
- [ ] 报告 TPOT p50/p95/p99。
- [ ] 报告 end-to-end latency 和 queue wait 分位数。
- [ ] 报告 input/output tokens/s 和 goodput。
- [ ] 调度决策自身 p95 延迟不成为 TTFT 主要瓶颈。
- [ ] token-aware/cache-aware 相对基线的提升必须在同硬件、同流量回放下比较。

### 12.3 资源效率

- [ ] 报告 GPU utilization、VRAM、KV cache utilization 和 prefix cache hit rate。
- [ ] 报告每 1K input/output token 的资源成本。
- [ ] 长短请求混合下没有持续饥饿；公平性指标有明确定义。
- [ ] 过载时内存、队列和连接数保持有界。

### 12.4 上下文质量

- [ ] 长对话事实召回、长文档 QA、定位、综合生成分别报告。
- [ ] 同预算公平 baseline 与完整上下文参考同时存在。
- [ ] 自动压缩/降级造成的质量损失低于预先定义阈值。
- [ ] 失败案例分类能够指导路由，而不是只输出总体平均分。

---

## 13. 必做故障演练

- [ ] 推理 worker 在排队阶段崩溃。
- [ ] 推理 worker 在 prefill 前崩溃。
- [ ] 推理 worker 在首 token 后崩溃。
- [ ] 客户端在排队、prefill、decode 三个阶段取消。
- [ ] GPU OOM 和 CUDA 错误。
- [ ] Worker 心跳停止但进程仍存活。
- [ ] Kubernetes API、Worker Registry或Temporal短暂不可用。
- [ ] Redis 短暂不可用或发生主从切换。
- [ ] Chat Service 在 reservation 创建后崩溃。
- [ ] 控制面与计算节点网络分区。
- [ ] 模型 revision 滚动升级。
- [ ] context-engine 不可用或超时。
- [ ] Prefix cache 版本不一致或租户不匹配。
- [ ] 单节点磁盘模型缓存损坏。
- [ ] 过载突发、慢客户端和不读取流的客户端。

每次演练必须保存：拓扑、命令、时间线、预期、实际结果、指标截图/导出、资源是否回收、遗留问题和修复 commit。

---

## 14. 证据包与文档交付

### 14.1 仓库内必须保留

- [ ] 架构图和关键 ADR。
- [ ] Model/Worker/Request/Reservation schema。
- [ ] 本地与多节点部署说明。
- [ ] API/错误码/降级语义。
- [ ] 运维和故障排查手册。
- [ ] benchmark manifest 与小型结果摘要。
- [ ] 负向实验和适用边界。
- [ ] 发布 checklist 和回滚步骤。

### 14.2 Evidence manifest 最低字段

```text
run_id
git_commit / git_dirty
command / config_hash
started_at / duration
host / os / container image digest
cpu / ram / gpu / driver / cuda
engine / engine_version
model_id / model_revision / tokenizer_revision
dataset / dataset_revision / sample_ids / seed
traffic_shape / concurrency / prompt-output distribution
raw_result_paths / summary / checksum
```

### 14.3 宣传与简历表述门禁

- 未完成 Phase 1–3：只能称“多节点推理路由原型”。
- 完成 Phase 1–4：可以称“资源感知推理控制面”，但必须说明验证拓扑。
- 完成 Phase 1–7 和第 12–14 节：才可以称“生产级小模型推理与调度平台”。
- “超长上下文”必须说明任务、模型、预算和策略；不得用定位任务准确率代表通用长上下文能力。
- “加速”必须说明是 prefill、decode、吞吐还是端到端延迟。
- 复用 vLLM/SGLang 的能力必须明确归属，自研部分聚焦控制面策略和集成闭环。

---

## 15. 代码组织建议

在不进行无必要大重构的前提下，建议逐步形成：

```text
services/
  api-gateway/
  chat-service/
  scheduler-service/          # admission、placement、reservation
  context-engine/             # context planner，不承担 GPU 调度
  llm-inference/              # worker adapter，复用 vLLM/HF/SGLang
pkg/
  proto/
  registry/
  scheduling/                 # 纯领域模型和可测试策略
deploy/
  compose/
    local-cpu/
    local-gpu/
    multi-node/
  observability/
docs/
  architecture/
  operations/
  evidence/
```

是否拆出独立 `scheduler-service` 必须由故障域、扩缩和所有权需求决定；在领域模型稳定前，可先保留在 Chat Service 内，避免为了微服务而微服务。

---

## 16. 实施顺序与依赖

严格顺序：

1. Phase 0 冻结基线和证据格式。
2. Phase 1 修复地址、注册、部署和 readiness。
3. Phase 2 建立可信资源观测。
4. Phase 3 才实现资源感知调度和 reservation。
5. Phase 4 补齐故障恢复和生命周期。
6. Phase 5 将上下文成本与 cache affinity 接入调度。
7. Phase 6 完成可观测性、安全和运维。
8. Phase 7 完成压测、canary 和发布验证。
9. 第 12–14 节最终验收、证据归档和表述审计。

禁止跳过资源观测直接编写复杂调度公式；没有真实状态输入的“智能调度”只会放大错误。

---

## 17. 每个 PR 的强制模板

每个实施 PR 必须回答：

1. 对应本计划哪个 checkbox？
2. 改变了哪个运行时契约或故障语义？
3. 正常路径和失败路径分别如何验证？
4. 使用了哪些成熟组件，哪些逻辑是项目自研？
5. 是否影响本地模式、多节点模式和向后兼容？
6. 新增了哪些指标、日志和告警？
7. 回滚方法是什么？
8. 证据文件或 CI run 在哪里？

PR 合并后只有在证据齐全时才能把对应项改成 `[x]`。

---

## 18. 最终 Definition of Done

项目只有同时满足以下条件，才算完成本计划：

- 本地 CPU、单 GPU、多 worker 单机和至少双节点拓扑均可重复部署。
- worker 注册的地址、模型和资源能力真实、可达、会过期、可 drain。
- 调度基于真实 token/GPU/KV/queue 状态，拥有可解释候选过滤和评分。
- reservation 在成功、失败、超时、取消和进程崩溃下都能最终回收。
- 过载时系统有界并提供明确的排队、降级或拒绝语义。
- 节点失败、网络分区、OOM、控制面故障和滚动升级均经过演练。
- 长上下文策略与模型窗口、KV 成本和缓存亲和性进入统一决策。
- 指标、trace、日志、告警和运维手册足以定位一次跨服务请求。
- 安全和租户隔离覆盖 API、节点通信、上下文与 prefix cache。
- 所有核心结论有带 provenance 的可复现实验，不依赖 README 自述。
- 生产路径不存在未标注 placeholder；mock-only 测试不被计作 E2E 证据。
- 项目负责人完成最终审查并明确允许删除本文件。

在此之前，本文件必须继续保留在仓库中，作为范围控制、验收和防止项目重新变成“堆功能目录”的约束。
