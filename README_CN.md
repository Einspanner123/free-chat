# Free Chat — LLM 工程平台

基于微服务的 LLM 应用开发平台，覆盖**对话服务、上下文管理、模型微调、RAG、评测**。Go 控制面 + Python 计算面。

---

## 项目定位

Free Chat 是一个 **LLM 应用平台**，而非推理引擎或训练框架。它位于 vLLM、PyTorch 等基础设施之上，将它们编排为应用可用的服务。

技术上最有差异化的部分是**上下文管理系统**——它解决一个真实的生产问题：如何在数百轮对话中保持 LLM 的连贯性，同时不超出上下文窗口限制或爆炸性增加推理成本。

---

## 上下文管理系统

聊天服务实现了一个多阶段 pipeline，在每一轮决定保留哪些上下文。

### Pipeline

```
用户消息 → 预算检查 (tiktoken 估算, ±3-5%)
        → 预算充足?  → 全量上下文，不压缩
        → 超预算?    → 按时间递减的分级压缩
        → 严重超预算? → 话题分析 → 用户选择焦点
        → 构建含 attention sink 优化的结构化上下文
        → 发送给推理引擎
```

### 分级压缩

当对话超出 token 预算时，消息不是简单地截断，而是根据距当前轮的远近分级压缩：

| 级别 | 范围 | 处理方式 |
|------|------|---------|
| 原文保留 | 最近 5 轮 | 完全保留 |
| 轻量压缩 | 第 6-20 轮 | 截断至 100 字符 |
| 中量压缩 | 第 21-50 轮 | 截断至 50 字符 |
| 重量压缩 | 51 轮以上 | 替换为 "[compressed]" |
| 丢弃 | 超出预算 | 移除 |

### 话题感知重建

当话题漂移且压缩不足以控制预算时，系统从历史中提取话题，让用户选择保留哪些。

### Attention Sink 缓解

```
位置 0:  "\n\n"                          ← sink token
位置 1:  System 指令                      ← 首位效应
位置 N:  对话历史                          ← 时间顺序
位置 N+1: System: 指令重申                 ← 近因效应
位置 N+2: 当前输入
```

### 效率

50 轮对话（约 12,847 tokens）在分级压缩下缩减到 3,824 tokens（70.2%），话题重建后进一步到 2,156 tokens（83.2%）。

---

## 推理组件

### 引擎后端

支持 HuggingFace Transformers 和 vLLM，通过 `ENGINE_TYPE` 环境变量选择。`BaseEngine` 接口定义了 `generate`、`stream_generate`、`count_tokens`、`get_metrics`。

### 量化

支持 AWQ、GPTQ、SqueezeLLM，通过 `QUANTIZATION` 环境变量选择。

### KV Cache 管理

基于 block 的 `KVCacheManager` 实现了 LRU/滑动窗口/注意力加权驱逐策略、prefix cache 以及引擎注入适配器。

### Continuous Batching Scheduler

迭代级调度器（Orca 风格），与 KV Cache Manager 联动，每步分配/释放 block。

### Speculative Decoding

草稿-验证循环，通过拒绝采样算法验证候选 token。

---

## LLM 生命周期模块

| 模块 | 功能 | 目录 |
|------|------|------|
| 微调 | LoRA/QLoRA，可配置 rank 和量化 | `services/finetune/` |
| 对齐 | DPO 偏好优化 | `services/alignment/` |
| RLHF | PPO 强化学习 | `services/rlhf/` |
| RAG | 文档分块、稠密/稀疏/混合检索 | `services/rag/` |
| 评测 | MMLU, C-Eval, GSM8K, HumanEval | `services/evaluation/` |
| 合成数据 | Self-instruct, 数据增强 | `services/synthetic-data/` |

---

## 测试覆盖

| 模块 | 测试数 | 范围 |
|------|--------|------|
| inference-engine | 68 | KV cache, 调度器, benchmark |
| llm-inference | 145 | 引擎后端, 量化, 优化 |
| finetune | 110 | LoRA/QLoRA 训练, 数据加载 |
| evaluation | 90 | MMLU/C-Eval/GSM8K/HumanEval |
| rag | 51 | 检索策略, 分块 |
| alignment | 50 | DPO 损失, 偏好数据 |
| synthetic-data | 38 | 数据生成, 质量过滤 |
| rlhf | 21 | PPO 损失, GAE 估计 |

总计: **573 tests**.
