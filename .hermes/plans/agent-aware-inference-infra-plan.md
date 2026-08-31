# FreeChat Agent-Aware Inference Infrastructure 完整实施计划

> 状态：ACTIVE / 上位实施与验收契约
> 适用分支：`main` 及后续开发分支
> 核心定位：面向 Agent/Harness 工作负载、运行在 Python/Triton 与推理引擎交界处的异构 GPU 模型执行、KV 管理、并行规划与调度系统。
> 删除规则：**只有本计划全部代码、测试、真实双机 GPU 验收和证据归档完成，并由项目负责人明确确认后，才允许独立提交删除。**

---

## 0. 文档治理

### 0.1 权威关系

- 本文件取代 `.hermes/plans/production-inference-platform-plan.md`，成为新增实现的唯一上位计划。
- 需求来源记录进入 `REFERENCE_ONLY`，仅用于决策溯源；在目标系统验收和负责人确认前不得删除。
- `.hermes/plans/context-management-plan.md` 仅作为历史设计资料，不再定义产品边界。
- 出现设计冲突时，必须创建 `CONFLICT` 状态，记录冲突内容、相关文件/提交、可选方案、影响和负责人回复；得到用户明确决定前不得自行越过冲突。
- 不允许用人力或日历时间缩减范围。功能边界由本计划的产品职责、非目标与验收门禁决定。

### 0.2 状态与完成口径

- `[ ]`：未开始或未验收。
- `[~]`：实现存在，但证据不完整，不计为完成。
- `[x]`：实现、测试、真实运行证据和文档均齐全。
- `[!]`：阻塞或冲突，必须记录解除条件。
- 单元测试、mock、配置文件、README 声明、接口返回 200 或一次演示都不能单独证明功能完成。
- 所有性能结论必须记录 commit、模型 ID/revision、引擎版本、CUDA/驱动、GPU、拓扑、dtype/量化、并发、输入/输出分布、重复次数和原始结果路径。
- 所有优化必须同时报告收益、退化区间、显存/CPU/网络代价与关闭开关。

### 0.3 删除门禁

删除本计划前必须满足：

- [ ] 第 7 节全部工作包完成。
- [ ] 第 8 节真实硬件与故障验收完成。
- [ ] 第 9 节 benchmark 原始数据、分析脚本和可复现实验清单已入库。
- [ ] 第 10 节迁移矩阵执行完毕，没有历史主路径暗中承担目标系统职责。
- [ ] 所有 P0/P1 问题关闭；P2 有明确影响说明与负责人决定。
- [ ] 没有 placeholder、mock-only、静态假指标或未运行的 GPU 测试冒充完成。
- [ ] README、架构图、部署方法和默认运行路径与代码一致。
- [ ] 项目负责人以 issue、PR 或提交说明明确写出“同意删除 Agent-Aware Inference Infrastructure 计划”。
- [ ] 删除使用独立提交，并引用最终证据索引。

---

## 1. 经纠正的项目定义

### 1.1 一句话定义

FreeChat 是一个 **harness-neutral、agent-aware 的分布式推理基础设施实验与生产化平台**：接收 Agent/Harness 暴露的会话、分支、暂停/恢复、优先级和前缀复用语义，把它们下沉为模型选择、批处理、抢占、KV 生命周期、worker placement、并行方案和 kernel/内存流水线决策。

### 1.2 项目所在层级

```text
Agent / Coding Harness
  OpenAI Agents SDK | LangGraph | AutoGen | custom loop | trace replay
                         │
                         │ OpenAI-compatible request + validated agent hints
                         ▼
FreeChat Agent-Aware Inference Infra
  workload model / router / scheduler / cache controller / parallel planner
                         │
                         │ engine adapter + cache/telemetry control
                         ▼
Inference Runtime
  vLLM | SGLang | optional Dynamo interoperability
                         │
                         │ PyTorch / Triton / NCCL / CUDA runtime
                         ▼
GPU / CPU pinned memory / NVMe / network topology
```

FreeChat 不接管 Agent loop。Harness 决定提示、工具、handoff、subagent 和业务状态；FreeChat 只消费经过验证的 workload hints，并拥有最终资源决策权。

### 1.3 为什么 Agent/Harness 改变推理 infra

普通在线聊天常被近似为独立请求；Agent workload 具有不同结构：

- 同一 session 反复携带单调增长的公共前缀，呈现 write-once/read-many KV 模式。
- 工具调用造成秒级到分钟级空闲，恢复时仍要求低 TTFT；普通 LRU 会在暂停期间误淘汰高价值前缀。
- lead agent、subagent、重试和分支共享大段 system prompt、tool schema 与历史，但生命周期和复用概率不同。
- 一次用户任务形成多次短 decode、长 prefill、fork/join 和 burst，而不是单一稳定 QPS。
- Harness 可能同时调用本地小模型、远端高阶模型、draft model 或 verifier，形成异构模型与不同 SLO。
- 取消、审批、工具失败和恢复是正常路径，资源必须可抢占、释放并可解释地恢复。

因此优化目标不只是单请求 tok/s，而是 **每个 agent task 的总完成时间、有效 token 计算量、KV 复用率、GPU 成本和尾延迟**。

### 1.4 核心差异化

项目不得以“支持 Agent”或“prefix-cache aware”作为独创卖点。相邻系统已覆盖通用 KV-aware routing、PD disaggregation 和多级 KV cache。本项目的可辩护差异化必须由证据支撑：

1. 在消费级、异构显存、无 NVLink/RDMA、低速或抖动局域网条件下，做正确的 placement 与 recompute/transfer 选择。
2. 把 harness 的 branch、pause、resume、lifecycle 和 predicted reuse 转化为可审计的 cache admission、retention、eviction 和 prefetch 策略。
3. 让模型架构特征与硬件拓扑共同参与执行计划：MHA/GQA/MLA、Dense/MoE、KV dtype、上下文长度、TP/PP/EP/CP/DP 与 PD 拆分。
4. 在 Python/Triton 层实现经过 profiler 证明的 hot-path 优化，并给出数值正确性、微基准和端到端 agent trace 收益。
5. 提供可复现的 agent workload trace/replay 和反事实策略比较，而不是只展示 WebUI 聊天成功。

### 1.5 非目标

- 不自研通用 Agent framework、planner、tool runtime、MCP server、A2A control plane 或长期业务记忆产品。
- 不重新实现完整 vLLM/SGLang/Dynamo，不复制已有 PagedAttention、FlashAttention、continuous batching 或集群编排。
- 不把 Temporal、Kubernetes 微服务数量、PostgreSQL schema 或 WebUI 复杂度当作技术主线。
- 不以跨弱网络 TP/PP 演示冒充有效并行；不可行配置必须被 planner 拒绝或标为实验。
- 第一阶段不写手工 C/C++/CUDA kernel；优先 Python、PyTorch、Triton、现有 engine extension point。只有 Triton 无法实现且证据证明必要时，另立 ADR 请求扩边界。
- 不宣称“无限上下文”；长上下文必须以 KV 容量、压缩/卸载、准确性与恢复成本量化。
- 不补齐现有 placeholder RLHF、Reward Model、MMLU 等目录作为主线。

---

## 2. 真实验收环境与拓扑边界

### 2.1 当前硬件

```text
ross
  Ubuntu 24.04
  RTX A6000 48 GiB
  125 GiB RAM
  1 GbE LAN + Tailscale

workstation
  Ubuntu 22.04
  RTX A5000 24 GiB + RTX A4000 16 GiB
  30 GiB RAM
  两卡无 NVLink，跨卡路径不视为高速互联
  Wi-Fi/LAN + Tailscale，当前可能经 DERP
```

### 2.2 当前必须诚实接受的结论

- 三张 GPU 默认作为独立 worker，跨节点采用 data/request parallel 与 session affinity。
- ross 与 workstation 之间默认禁止 TP、PP、EP collective 和跨机 PD KV 热迁移；当前链路不足以支撑其生产收益假设。
- workstation 双卡 TP/PP 只进入实验矩阵，必须先测 P2P、PCIe 带宽、通信占比与端到端收益；不因“能启动”进入默认路径。
- A6000/A5000/A4000 均为 Ampere。kernel 优化必须基于实际 SM、显存带宽、Tensor Core dtype 支持和 profiler 数据，不照搬 Hopper/Blackwell 假设。
- 多级 KV 首先验证 GPU HBM → pinned CPU RAM；NVMe 层在磁盘与文件系统基线合格后启用。跨节点共享 KV 仅作为未来高速网络能力，不列入当前 P0。

### 2.3 面向未来的拓扑能力模型

Worker 必须上报并周期校验：

- GPU 型号、compute capability、SM 数、显存总量/空闲量、支持 dtype。
- GPU-GPU P2P、PCIe generation/width、NVLink/NVSwitch、NCCL 实测矩阵。
- CPU NUMA、pinned memory 容量、GPU 到 NUMA 节点距离。
- 本地 NVMe 顺序/随机带宽与延迟。
- 节点间 RTT、抖动、有效带宽、丢包、RDMA/NIXL 能力。
- engine、model、tokenizer、KV block layout 和 quantization compatibility。

Parallel Planner 只能从能力模型中选择方案；不得用静态主机名分支硬编码。

---

## 3. Agent Workload Contract

### 3.1 北向兼容接口

- 保留 `/v1/chat/completions`，并实现 `/v1/responses` 所需的流式、工具调用和 continuation 语义。
- Agent 信息通过版本化扩展 `agent_hints` 和标准 header 传入；没有 hints 的普通客户端仍可运行。
- 提供 Python SDK/middleware，首批适配 OpenAI Agents SDK、LangGraph 和一个最小 custom agent loop。
- Harness adapter 只做字段映射、trace 关联和生命周期通知，不侵入框架 planner/tool 实现。

### 3.2 版本化 hints schema

最小字段：

```text
schema_version
harness_id / harness_version
task_id / session_id / agent_id / parent_agent_id
branch_id / parent_branch_id / turn_id / call_id
lifecycle: spawn | active | tool_wait | resume | terminal | cancelled
prefix_scope: global | harness | task | agent | branch | private
reuse_class: immutable_shared | growing_history | ephemeral_reasoning | unknown
expected_reuse_probability / expected_resume_ms / ttl_ms
priority / deadline_ms / expected_output_tokens
privacy_domain / tenant_id
allow_preemption / allow_kv_offload / allow_remote_worker
```

### 3.3 信任边界

- hints 是不可信建议，不是资源命令。服务端必须验证类型、范围、租户权限和状态转换。
- 客户端不得指定物理 worker、强占 GPU、读取其他租户 cache 或伪造 cache key。
- server 生成 canonical task/session/branch identity，并记录原始 hints、归一化结果和拒绝原因。
- 无效 hints 不得静默影响调度；按策略明确忽略、降级或返回结构化错误。
- cache key 必须包含模型 revision、tokenizer、adapter/LoRA、chat template、tenant/privacy domain 与 token block hash。

### 3.4 生命周期状态机

```text
NEW → ACTIVE → TOOL_WAIT → RESUMING → ACTIVE → TERMINAL
             ↘ CANCELLED         ↘ FAILED
```

- 每次迁移必须幂等、可追踪，并关联 trace span。
- `TOOL_WAIT` 释放 decode slot 和 request reservation，但不等于立即淘汰 KV。
- `TERMINAL/CANCELLED` 触发 lifecycle-aware cache value 下调；共享不可变前缀不随单个 agent 一并删除。
- 迟到、重复和乱序事件必须有确定性处理规则和测试。

---

## 4. 目标架构

### 4.1 组件

```text
Gateway / Harness Adapters
  ├─ protocol normalization + structured streaming parser
  └─ hints validator + lifecycle ingest

Trace & Replay Plane
  ├─ request/token/cache/topology trace
  ├─ anonymizer + immutable manifest
  └─ open-loop / closed-loop / lifecycle replay

Agent-Aware Scheduler
  ├─ admission + priority + preemption
  ├─ prefix/cache affinity routing
  ├─ pause/resume and fork-group policy
  └─ model/worker/parallel-plan selection

KV Control Plane
  ├─ block lineage and value estimator
  ├─ admission/retention/eviction/prefetch policy
  ├─ HBM/CPU/NVMe placement catalog
  └─ recompute-vs-transfer cost oracle

Execution & Parallel Planner
  ├─ model architecture descriptor
  ├─ topology capability graph
  ├─ DP/TP/PP/EP/CP/PD candidate generator
  └─ measured cost model + safe fallback

Engine Adapters
  ├─ vLLM
  ├─ SGLang
  └─ optional Dynamo interoperability/reference path

Kernel Lab
  ├─ torch.profiler / Nsight evidence
  ├─ Triton/PyTorch implementations
  ├─ correctness + microbench
  └─ engine integration + end-to-end ablation
```

### 4.2 控制路径与数据路径

- Python 负责策略、模型描述、cost model、trace replay、实验编排和 SDK。
- 请求热路径避免同步数据库访问；配置、cache catalog 和 topology snapshot 使用内存快照与版本号。
- 大块 KV 数据不经过 Python 对象序列化或 PostgreSQL/Redis；使用 engine connector、pinned memory、共享内存或已验证的传输库。
- engine adapter 通过 capability API 暴露真实 cache block、queue、preemption、token 与显存指标；缺失能力必须显式标记，不伪造。
- 策略输出包含 decision ID、输入 snapshot、候选方案、cost 分解、选中原因和 fallback。

### 4.3 复用边界

- vLLM/SGLang：模型执行、attention kernel、continuous batching、基础 prefix cache。
- LMCache/HiCache/KVBM 等：作为 KV 分层/传输 provider 候选，通过适配与基准选择，不复制其完整实现。
- NVIDIA Dynamo：作为相邻系统、互操作目标和 baseline；不把已有 agent hints、KV-aware router 或 PD disaggregation 改名重做。
- PyTorch/Triton：kernel 原型、编译、autotune 和 correctness reference。
- OpenTelemetry + Prometheus：trace/metrics 标准，不自研监控存储。
- PostgreSQL 可保存实验 manifest、决策与审计；Redis 仅在经基准证明有必要时保存短期租约，不进入 token/KV 热路径。

---

## 5. 调度、缓存和并行算法边界

### 5.1 优化目标

每个候选 placement/plan 的目标函数至少包含：

```text
cost = predicted_queue_delay
     + predicted_prefill_time
     + predicted_decode_time
     + kv_load_or_transfer_time
     + communication_time
     + cold_start_penalty
     + deadline_risk_penalty
     + cache_eviction_externality
```

硬约束先过滤：模型/格式兼容、显存、上下文容量、privacy、deadline 可行性、拓扑能力、worker 健康。软目标只能在可行集合中排序。

### 5.2 Agent-aware KV value

cache block value 是可解释的策略函数，而非神秘分数，至少考虑：

- prefix 类型与共享范围。
- 当前引用 agent/branch 数。
- 最近复用与复用间隔分布。
- harness 给出的 resume 预测及其历史校准误差。
- 重算 token 数与对应 GPU prefill 成本。
- 从 CPU/NVMe 恢复的实测成本。
- 占用 HBM 导致其他请求被拒绝或抢占的机会成本。
- 生命周期：active、tool wait、terminal、cancelled。
- tenant/privacy 隔离和 TTL。

首版先实现 LRU、prefix-affinity、cost-aware、lifecycle-aware 四种可切换策略，并用同一 trace 反事实比较。

### 5.3 Fork/Join 与共享前缀

- 公共 token blocks 采用不可变内容寻址和引用计数；branch 只记录增量 lineage。
- subagent 创建不得复制整段 Python token/KV 对象。
- branch 终止只释放私有后缀引用；共享系统提示和工具定义按全局价值策略保留。
- 必须处理不同 chat template、tool schema 顺序、tokenizer revision 导致的 cache miss，不允许字符串“看起来相同”就共享。

### 5.4 Pause/Resume

- `TOOL_WAIT` 时 scheduler 估计 `keep HBM`、`offload CPU`、`offload NVMe`、`evict/recompute` 四种成本。
- `expected_resume_ms` 只能作为特征；必须用实际恢复分布持续校准。
- resume 前验证 worker/model/cache generation；cache 不可用时自动重算并记录原因，不能返回错误结果。
- prefetch 使用独立 stream 与 pinned buffer，并量化是否遮蔽在排队/前处理时间内。

### 5.5 并行策略

Planner 的候选维度：

- DP/request parallel：默认跨节点方案。
- TP：仅在同节点、P2P/NVLink/PCIe 实测和模型切分收益可行时启用。
- PP：考虑 stage memory、activation transfer、microbatch 和 agent 短 decode bubble；弱网默认拒绝。
- EP：仅对 MoE 模型，必须考虑 all-to-all、expert imbalance 和 token routing。
- CP/sequence parallel：仅在长上下文显存成为硬约束且通信可承受时评估。
- PD disaggregation：prefill/decode 资源形态差异显著且 KV transfer 路径足够快时评估。
- Speculative decoding：考虑 draft/target placement、acceptance rate、额外 KV 和 agent 输出长度分布。

静态规则提供安全 fallback；learned/autotuned cost model 不得绕过硬约束。

### 5.6 模型架构描述

Model Descriptor 至少记录：

- layer、hidden、head、KV head、head dimension、vocab、max position。
- MHA/GQA/MQA/MLA attention 类型与 KV 表示。
- Dense/MoE、expert 数、top-k、shared expert。
- RoPE/scaling、sliding window、chunked prefill 支持。
- dtype/quantization、KV dtype、LoRA/adapter。
- tool/reasoning parser、speculative compatibility。
- 实测 memory profile、prefill/decode roofline 特征和 engine capability。

---

## 6. Kernel 与芯片感知工作边界

### 6.1 方法

每个 kernel 候选必须经历：

1. 用真实 agent trace 在 engine 端到端运行中定位瓶颈。
2. 用 torch.profiler 与 Nsight Systems/Compute 区分 compute、memory、launch、sync、PCIe 和网络瓶颈。
3. 建立 PyTorch/reference 实现和数值容差。
4. 实现 Triton 或 torch.compile/custom-op 版本。
5. 覆盖 shape、dtype、alignment、长短序列和 OOM/fallback。
6. 做 microbenchmark，并回到端到端 trace 测收益。
7. 若端到端无稳定收益，保留实验报告但默认关闭，不包装成生产优化。

### 6.2 首批候选而非预设结论

- 多级 KV offload 的 pack/unpack、布局转换和量化/反量化融合。
- pinned CPU → GPU 的异步 block gather、prefetch 与 stream overlap。
- branch resume 场景的 block-table gather/scatter 或 copy-on-write 元数据更新热点。
- 针对当前模型架构的 RoPE + KV write、decode attention 或 MoE token dispatch 融合缺口。
- speculative decoding 中 verification/acceptance 的小 batch launch overhead。

已有 engine kernel 性能更好时直接复用。候选列表不构成必须自研；**必须实现的是 profiler 驱动的选择流程、至少一个经真实端到端验证的 hot-path 优化，或以完整证据证明当前硬件不存在值得替换的 kernel 并转向同层级的内存流水线优化。**

### 6.3 芯片理解的可交付证据

- 每张 GPU 的简化 roofline：峰值/实测算力、HBM 带宽、PCIe 带宽、kernel operational intensity。
- warp/CTA 划分、occupancy、register/shared-memory 使用与访存合并分析。
- prefill 与 decode 在不同 batch/context 下 compute-bound 或 memory-bound 的转折点。
- GQA/MoE/quantization 对 KV 容量、带宽和并行通信的影响。
- stream、event、pinned memory 与 overlap 时间线。
- 不要求手写 C，但必须能解释 Python/Triton 实现最终映射到的硬件执行行为。

---

## 7. 实施工作包

### WP0：现状冻结与可复现基线

- [ ] 冻结当前 commit、依赖、模型 revision 和两台服务器环境快照。
- [ ] 把 placeholder RLHF/Reward/MMLU 与 mock-only 能力从产品清单隔离。
- [ ] 建立直接 vLLM/SGLang 单 worker 基线，不经过 FreeChat 调度器。
- [ ] 建立 round-robin、least-load、prefix-affinity 三种路由基线。
- [ ] 采集 agent trace：长会话、工具等待、fork/subagent、取消/恢复、混合本地/远端模型。
- [ ] 原始 trace 匿名化、版本化并带 immutable manifest；禁止仅保留汇总图。

验收：一条命令可重放固定 trace 并产出带环境指纹的原始结果；重复运行方差有解释。

### WP1：协议与 Harness 适配

- [ ] 定义并版本化 `agent_hints` JSON Schema、header 和错误码。
- [ ] 完成 OpenAI-compatible Chat Completions 与 Responses 流式契约测试。
- [ ] OpenAI Agents SDK middleware。
- [ ] LangGraph middleware。
- [ ] 最小 custom loop 示例，证明协议不绑定某框架。
- [ ] lifecycle 乱序、重复、丢失和恶意 hints 测试。

验收：三个 harness 对同一逻辑 workload 产生统一 canonical trace；无 hints 客户端不退化。

### WP2：Trace、Replay 与可观测性

- [ ] 定义 task/session/agent/branch/call/token/cache-block span 关系。
- [ ] 实现 open-loop、closed-loop 和带 tool wait 的 lifecycle replay。
- [ ] 记录 queue、prefill、decode、tool wait、resume、cache lookup/load/evict 和网络时间。
- [ ] 支持同一 trace 对多策略反事实回放。
- [ ] Prometheus dashboard 和最小 WebUI 展示拓扑、运行 trace、KV 层级、调度原因与 benchmark。

验收：任一尾延迟样本可从 task 下钻到 worker、cache block 事件和 engine 指标。

### WP3：Agent-Aware Scheduler

- [ ] capability filter、租约、准入、背压、取消和幂等释放。
- [ ] LRU、least-load、prefix-affinity、cost-aware、lifecycle-aware 策略插件。
- [ ] fork-group placement、tool-wait 降权、resume 提升与 deadline/priority。
- [ ] recompute-vs-transfer cost oracle。
- [ ] 决策日志包含候选、过滤原因、score 分解和 fallback。
- [ ] worker crash、stale telemetry、网络分区和控制面重启恢复。

验收：在固定 agent trace 上，相对基线给出显著性、置信区间和无收益/退化场景；零资源泄漏。

### WP4：多级 KV 生命周期

- [ ] 先对接 engine 原生 prefix cache，获得真实 block/命中/驱逐 telemetry。
- [ ] 内容寻址、lineage、引用计数、tenant/privacy 隔离。
- [ ] HBM ↔ pinned CPU offload/load；NVMe 在基线通过后启用。
- [ ] keep/offload/evict/prefetch 策略与 lifecycle event 联动。
- [ ] 模型 revision、tokenizer、template、LoRA 与 KV layout 不兼容保护。
- [ ] crash 后 metadata 与实际 cache 不一致的安全恢复。

验收：工具等待和 subagent trace 中减少重算 token 或降低 task E2E；数据隔离、hash collision、防陈旧读取测试通过。

### WP5：模型与并行计划

- [ ] Model Descriptor 与 Topology Capability Graph。
- [ ] DP/TP/PP/EP/CP/PD/speculative 候选生成及硬约束过滤。
- [ ] 单 GPU memory/performance profiler 与离线 plan simulator。
- [ ] workstation 双卡实验，报告 TP/PP 的通信占比和拒绝/选择依据。
- [ ] 弱网络跨机 parallel plan 必须被拒绝；独立 worker 路由正常。
- [ ] 为未来 NVLink/RDMA fixture 提供可注入 topology profile 与集成测试。

验收：planner 对每个选择给出可复核 cost breakdown；预测误差在基线后设阈值并有回退。

### WP6：Kernel 与内存流水线

- [ ] 完成 A6000/A5000/A4000 roofline 与 engine profile。
- [ ] 从真实 hot path 选择候选，不预设“自研一定更快”。
- [ ] reference、Triton/custom-op、autotune cache、correctness 和 fallback。
- [ ] microbenchmark 覆盖目标 shape/dtype，报告 p50/p95 与显存。
- [ ] engine adapter 集成，运行 agent trace A/B。
- [ ] Nsight 时间线证明 overlap、launch 或 bandwidth 改善来自所述机制。

验收：至少一项默认可关闭的优化通过正确性与端到端门禁；若全部候选无收益，提交完整负结果与选择下一层瓶颈的证据，不得伪造亮点。

### WP7：生产边界与部署

- [ ] Docker Compose 双机最小部署和清理脚本。
- [ ] worker 身份、mTLS/API auth、tenant/cache 隔离与 secret 管理。
- [ ] readiness 只在模型真正可服务时通过。
- [ ] OOM、worker kill、router restart、tool wait 超时、网络抖动和磁盘耗尽演练。
- [ ] 配置 schema、升级/回滚、兼容矩阵和 runbook。
- [ ] WebUI 保持简单，只承担可使用性和证据展示，不演变成聊天产品主线。

验收：真实 ross + workstation 连续运行、故障注入和恢复证据齐全；重启后无幽灵租约、错误 KV 命中或静默请求丢失。

### WP8：开源与简历证据

- [ ] README 明确与 vLLM/SGLang/Dynamo/LMCache 的复用和差异，不宣称他人成果。
- [ ] 架构图、harness adapter 示例、策略扩展指南、kernel 实验指南。
- [ ] 一键 demo：运行最小 agent → tool wait → resume → trace/KV 决策可视化。
- [ ] 一键 benchmark：基线与优化策略对比，原始数据可下载。
- [ ] 建立 claims ledger：每条简历/README 性能声明链接到 commit、配置和原始证据。

验收：新的评审者能在干净环境复现 demo；面试问到任一数字时能展示实现路径、基线、失败案例和原始结果。

---

## 8. 验收矩阵与指标

### 8.1 指标

请求级：

- TTFT、ITL、E2E latency、output tok/s、goodput、取消响应时间。
- queue/prefill/decode/cache-load/transfer 分解。

Agent task 级：

- task completion time、每任务模型调用数、总 input/output/computed tokens。
- prefix cache hit tokens、recomputed tokens、resume TTFT。
- fork 冷启动成本、tool-wait 后保留收益、deadline miss rate。
- 每成功任务 GPU-seconds、HBM byte-seconds、CPU/NVMe traffic。

系统级：

- GPU util、SM occupancy、HBM/PCIe/网络带宽、KV tier occupancy。
- preemption、eviction、offload、prefetch hit、OOM、fallback、worker migration。
- scheduler prediction error 和 hint calibration error。

### 8.2 Workload cells

- 单普通聊天请求：确保 agent-aware 路径不造成不可接受开销。
- 1000 轮单调增长 session replay。
- 2–30 秒工具等待的 pause/resume 分布。
- lead + 4 subagent 共享前缀和独立分支。
- 短输出工具调用循环与长 reasoning 输出。
- 0.5B/3B/7B 级本地模型和一个 OpenAI-compatible 远端 provider。
- 单 GPU、workstation 双 GPU、三独立 worker、网络抖动/断连。
- cache 冷/热、HBM 压力、CPU offload、可选 NVMe。

### 8.3 基线与阈值原则

- 先建立直接 engine、round-robin、least-load、prefix-affinity 基线，再冻结数值阈值。
- 阈值必须按 workload cell 设置，不能用平均值掩盖长上下文或尾延迟退化。
- 默认启用条件：正确性无退化；目标 cell 有统计稳定收益；非目标 cell 的开销在冻结 guardrail 内；可快速关闭。
- 若当前硬件不支持某方案，验收结果应是“被 capability filter 正确拒绝”，而不是强行运行。

### 8.4 故障与安全

- worker 在 prefill/decode/offload 中途退出。
- lifecycle event 重复、乱序和断连后补发。
- cache catalog 存在但 block 已丢失；block 存在但 metadata 已过期。
- tokenizer/model/LoRA/template 更新后旧 KV 仍在。
- tenant 试图引用其他 tenant prefix。
- CPU pinned memory、NVMe、GPU HBM 达到限额。
- 网络由直连切换为 DERP、高抖动或分区。
- scheduler/control plane 重启后租约和请求状态恢复。

所有故障必须得到正确结果、明确失败或可解释 fallback；禁止静默错误复用。

---

## 9. 证据目录与结果治理

建议结构：

```text
benchmarks/
  traces/
  manifests/
  workloads/
  baselines/
  analysis/
  results/<date>/<commit>/<run-id>/
experiments/
  kernels/
  parallelism/
  cache-policy/
docs/
  architecture/
  decisions/
  runbooks/
  claims-ledger.md
```

- benchmark 结果不得全部 gitignore；至少提交小型原始样例、manifest、分析输出和大文件获取方法。
- 图表由仓库脚本从原始数据生成，不手工录入关键数值。
- 每次 run 包含 stdout/stderr、resolved config、environment fingerprint、GPU topology、metrics 和失败样本。
- 只允许在 claims ledger 标记为 `VERIFIED` 的数字进入 README 或简历。
- 设计候选标记 `PROPOSED`，实现未验收标记 `IMPLEMENTED_UNVERIFIED`，负结果标记 `REJECTED_BY_EVIDENCE`。

---

## 10. 现有项目迁移矩阵

### 10.1 选择性继承

- OpenAI-compatible API 代码：抽离为 Gateway/协议兼容层，补齐契约测试。
- vLLM/HuggingFace worker：转为 engine adapter 或最小 reference worker。
- Redis least-active/prefix-aware 概念：只作为 baseline，不能直接视为新 scheduler。
- tracing/metrics 基础：映射到 task/session/agent/branch/cache-block 模型。
- Docker 与部署脚本：修复后作为最小双机验收入口。

### 10.2 需要重写或降级

- ChatService 顶层业务编排：降级为 demo harness，不再拥有资源策略。
- context-engine 的长期事实记忆：移出 infra 主线；只保留对推理前缀和 trace 有用的最小 session fixture。
- Go/Python gRPC 链路：逐段基准；无必要的跨语言 hop 删除或移入 legacy。
- Consul/RocketMQ/Temporal/PostgreSQL/pgvector：只有新职责有真实需要才保留；不得因为已经存在而继续成为架构中心。
- WebUI：收缩为 topology、trace、cache、benchmark 和最小 playground 五个页面。

### 10.3 停止扩展

- placeholder RLHF、Reward Model、MMLU/evaluation façade。
- README 中未被运行证据支持的“生产级”“超长上下文”“智能调度”声明。
- 固定 last-10 历史与伪无限会话路径。
- 跨弱网 tensor/pipeline parallel 和 KV migration 的展示性代码。

### 10.4 迁移安全

- 旧文件先冻结、标记和建立映射，不直接批量删除。
- 目标基线运行后再迁移单个组件；每次迁移必须有 parity/fallback。
- 只有目标路径通过真实验收、引用清理和负责人确认后，才删除历史实现。
- 所有删除单独提交，列出删除对象、替代物、验证结果和恢复方式。

---

## 11. 建议仓库结构

```text
free-chat/
  gateway/
    openai_api/
    hints_schema/
    streaming_parser/
  harness_adapters/
    openai_agents/
    langgraph/
    custom_loop/
  scheduler/
    admission/
    policies/
    cost_model/
    leases/
  kv_control/
    lineage/
    policy/
    tiers/
    providers/
  planner/
    model_descriptor/
    topology/
    parallelism/
  engines/
    vllm/
    sglang/
    dynamo/
  kernels/
    reference/
    triton/
    benchmarks/
  tracing/
    schema/
    collector/
    replay/
  benchmarks/
  webui/
  deploy/
  docs/
  legacy/
```

目录按迁移逐步建立，不做一次性空目录脚手架。每个组件必须有明确 owner contract 和 failure semantics。

---

## 12. 实施顺序与阶段门禁

### Gate A：方向与基线

- 旧计划标记 superseded；新定义、非目标、claims boundary 经负责人确认。
- 完成 WP0，得到真实 agent trace 和 engine baseline。
- 未通过前禁止声称 agent-aware 优化有效。

### Gate B：语义进入系统

- 完成 WP1–WP2，harness lifecycle 可重放、可观察、可归一化。
- 未通过前不得开始基于猜测的 KV 智能策略。

### Gate C：策略闭环

- 完成 WP3–WP4，hints → decision → engine/cache action → metric → replay 闭环成立。
- 必须先以基线设阈值，再决定默认策略。

### Gate D：芯片与并行证据

- 完成 WP5–WP6，给出当前拓扑的选择/拒绝依据和至少一个真实 hot-path 结论。
- 不能用 microbenchmark 单点加速替代 task E2E 收益。

### Gate E：生产与交付

- 完成 WP7–WP8 和第 8 节故障矩阵。
- README/简历只引用 claims ledger 的 verified 项。
- 经负责人最终确认后，才处理计划文件删除。

---

## 13. 最终演示必须回答的问题

演示不是“聊天页面能回复”，而必须让评审者看到：

1. 同一 Agent task 如何产生多次模型调用、工具等待和 subagent 分支？
2. Harness 暴露了哪些 hints，服务端接受、修正或拒绝了什么？
3. Scheduler 为什么选择该模型、worker 和并行方案？其他候选为何被过滤？
4. 哪些 token 命中已有 KV，哪些被重算，哪些 block 在 HBM/CPU/NVMe？
5. 工具等待期间为什么保留、卸载或淘汰某前缀？恢复成本是否符合预测？
6. 当前 A6000/A5000/A4000 与网络拓扑为什么选择独立 worker，而非跨机 TP/PP？
7. kernel/内存流水线优化改变了哪个硬件瓶颈，端到端收益在哪里失效？
8. worker crash、网络抖动或 cache metadata 陈旧时如何保证不产生错误结果？
9. 相比直接 vLLM/SGLang、简单 prefix affinity 和 Dynamo 相邻能力，本项目新增了什么可复现证据？

只有上述问题能由代码、trace、原始 benchmark 和故障证据共同回答，项目才具备大厂系统/推理 infra 简历项目的可信度。

---

## 14. 当前冻结决策

- 项目顶层从聊天/Agent 应用控制面转为 Agent-Aware Inference Infra。
- Harness 负责 agent loop；FreeChat 不接管工具和 planner。
- 主实现层为 Python/PyTorch/Triton + engine adapters；不以 C/C++/CUDA 为首阶段要求。
- 复用成熟 engine/cache/telemetry；自研集中在 workload contract、异构拓扑 cost model、KV lifecycle policy、parallel planner 和经证据选择的 hot path。
- 当前双机弱网络默认独立 worker；跨节点 TP/PP/EP/PD 热迁移不是 P0。
- WebUI 必须有，但只服务于可使用性、trace、拓扑、cache 和 benchmark 展示。
- 所有冲突进入专门状态并询问负责人，不由实现者擅自选择。
- 所有阈值先建立基线再冻结。
- 工作量不因时间假设缩减，但不得越过本计划非目标伪造更大范围。

本计划仍允许技术选型被证据推翻。任何推翻必须通过 ADR 记录原假设、实验、结果、迁移影响和负责人决定；沉没成本不得成为保留错误架构的理由。
