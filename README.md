<div align="center">

# Free Chat — LLM Engineering Platform

<a href="https://github.com/Einspanner123/free-chat"><img src="https://img.shields.io/badge/GitHub-Free%20Chat-blue?logo=github"></a>

[⬇️ English](#english) · [⬇️ 中文](#chinese)

</div>

---

<a id="english"></a>

# Free Chat — LLM Engineering Platform

A microservices-based platform for LLM application development, covering **conversation serving, context management, model fine-tuning, RAG, and evaluation**. Built with Go (control plane) and Python (compute plane).

## What This Project Is

Free Chat is an **LLM application platform** that integrates the full lifecycle of deploying and customizing large language models. It is not an inference engine (like vLLM) or a training framework—it sits above those layers, orchestrating them for application use.

The most technically differentiated part is the **context management system**, which addresses a real production problem: how to keep LLM conversations coherent over hundreds of turns without exceeding context window limits or exploding inference cost.

## Context Management System

### Pipeline

```
User Message → Budget check (tiktoken estimation, ±3-5%)
            → Under budget?  → Full context, no compression
            → Over budget?   → Hierarchical compression by recency
            → Severely over? → Topic analysis → user selects focus
            → Build structured context with attention sink mitigation
            → Send to inference engine
```

### Hierarchical Compression

| Level | Range | Treatment |
|-------|-------|-----------|
| Verbatim | Last 5 turns | Full content preserved |
| Light | Turns 6-20 | Truncated to first 100 chars |
| Medium | Turns 21-50 | Truncated to first 50 chars |
| Heavy | Turns 51+ | Replaced with "[compressed]" |
| Discard | Beyond budget | Removed |

### Topic-Aware Reconstruction

When compression alone is insufficient and the conversation exceeds 3 turns, the system extracts topics from history and lets the user select which to retain.

### Efficiency

A 50-turn conversation (12,847 tokens) compresses to 3,824 tokens (70.2%) under tiered compression, and to 2,156 tokens (83.2%) after topic reconstruction.

## Inference Components

| Component | Description | Location |
|-----------|-------------|----------|
| Engine backends | HuggingFace Transformers, vLLM (selectable via ENGINE_TYPE) | `services/llm-inference/` |
| Quantization | AWQ, GPTQ, SqueezeLLM (selectable via QUANTIZATION) | `services/llm-inference/` |
| KV Cache Manager | Block-based pool, LRU/sliding window/attention-weighted eviction | `inference-engine/memory-manager/` |
| Continuous Batching | Iteration-level scheduling with KVCache integration | `inference-engine/scheduler/` |
| Speculative Decoding | Draft-verify loop with rejection sampling | `services/llm-inference/src/optimization/` |

## LLM Lifecycle Modules

| Module | Function | Directory |
|--------|----------|-----------|
| Fine-tuning | LoRA/QLoRA with configurable rank, target modules, quantization | `services/finetune/` |
| Alignment | DPO preference optimization | `services/alignment/` |
| RLHF | PPO-based reinforcement learning from human feedback | `services/rlhf/` |
| RAG | Document chunking, dense/sparse/hybrid retrieval | `services/rag/` |
| Evaluation | MMLU, C-Eval, GSM8K, HumanEval benchmarks | `services/evaluation/` |
| Synthetic Data | Self-instruct, evol-question, EDA augmentation | `services/synthetic-data/` |

## Experiments

```
experiments/
├── context_compression/    # Full Context vs Truncation vs Hierarchical Compression
├── quantization/           # FP16 vs INT8 vs GPTQ vs AWQ
├── kv_cache/               # No Cache vs KV Cache vs Prefix Cache
└── speculative_decoding/   # γ=3/5/7, acceptance rate, speedup
```

Each experiment can be run with: `python experiments/<name>/run.py`

## Test Coverage

| Module | Tests | Scope |
|--------|-------|-------|
| inference-engine | 68 | KV cache, scheduler, benchmarks |
| llm-inference | 145 | Engine backends, quantization, optimizations |
| finetune | 110 | LoRA/QLoRA training, data loading, merging |
| evaluation | 90 | MMLU/C-Eval/GSM8K/HumanEval execution |
| rag | 51 | Retrieval strategies, chunking |
| alignment | 50 | DPO loss, preference data |
| synthetic-data | 38 | Generation, quality filtering |
| rlhf | 21 | PPO loss, GAE estimation |

Total: **573 tests**.

---

# 中文

<a id="chinese"></a>

# Free Chat — LLM 工程平台

基于微服务的 LLM 应用开发平台，覆盖**对话服务、上下文管理、模型微调、RAG、评测**。Go 控制面 + Python 计算面。

## 项目定位

Free Chat 是一个 **LLM 应用平台**，而非推理引擎或训练框架。它位于 vLLM、PyTorch 等基础设施之上，将它们编排为应用可用的服务。

技术上最有差异化的部分是**上下文管理系统**——它解决一个真实的生产问题：如何在数百轮对话中保持 LLM 的连贯性，同时不超出上下文窗口限制或爆炸性增加推理成本。

## 上下文管理系统

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

## 推理组件

| 组件 | 说明 | 位置 |
|------|------|------|
| 引擎后端 | HuggingFace Transformers / vLLM (ENGINE_TYPE) | `services/llm-inference/` |
| 量化 | AWQ, GPTQ, SqueezeLLM (QUANTIZATION) | `services/llm-inference/` |
| KV Cache 管理 | BlockPool + 三种驱逐策略 | `inference-engine/memory-manager/` |
| Continuous Batching | 迭代级调度 | `inference-engine/scheduler/` |
| Speculative Decoding | 草稿-验证循环 | `services/llm-inference/src/optimization/` |

## LLM 生命周期模块

| 模块 | 功能 | 目录 |
|------|------|------|
| 微调 | LoRA/QLoRA | `services/finetune/` |
| 对齐 | DPO 偏好优化 | `services/alignment/` |
| RLHF | PPO 强化学习 | `services/rlhf/` |
| RAG | 文档分块、稠密/稀疏/混合检索 | `services/rag/` |
| 评测 | MMLU, C-Eval, GSM8K, HumanEval | `services/evaluation/` |
| 合成数据 | Self-instruct, 数据增强 | `services/synthetic-data/` |

## 实验

```
experiments/
├── context_compression/    # 全量上下文 vs 截断 vs 分级压缩
├── quantization/           # FP16 vs INT8 vs GPTQ vs AWQ
├── kv_cache/               # 无缓存 vs KV Cache vs Prefix Cache
└── speculative_decoding/   # γ=3/5/7, 接受率, 加速比
```

运行: `python experiments/<name>/run.py`

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

总计: **573 个测试**。
