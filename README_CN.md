[English](README.md) | [中文](README_CN.md)

# Free Chat -- 面向小模型的长上下文框架

通过上下文管理与 RAG 检索，扩展小语言模型（0.5B-3B）的有效上下文长度。Go 控制面，Python 计算面。

---

## 概述

项目分为两层：

- **应用层**（`services/`）：可运行的微服务（聊天、认证、网关、推理、RAG、微调、评测）。
- **研究层**（`research/`）：在真实硬件上验证上下文框架的 benchmark 与实验。

核心工作是 **context-engine**：分层管道（检索 → 压缩 → 布局），在 token 预算内为小模型准备优化上下文。

---

## 项目结构

```
services/                    # 应用层
├── api-gateway/             # HTTP 网关 (Go)
├── auth-service/            # 用户认证 (Go)
├── chat-service/            # 对话服务，含上下文管理 (Go)
│   └── internal/interfaces/context_client.go  # 调用 context-engine 的 gRPC 客户端
├── llm-inference/           # 推理引擎：HF/vLLM 后端 (Python)
├── context-engine/          # 上下文优化：strategies/retriever/pipeline + gRPC 服务
├── rag/                     # RAG：分块、embedding、BM25/稠密/混合检索
├── finetune/                # LoRA/QLoRA 微调
├── alignment/               # DPO 偏好对齐
├── rlhf/                    # PPO RLHF
├── evaluation/              # MMLU, C-Eval, GSM8K, HumanEval
└── synthetic-data/          # Self-instruct, 数据增强

research/                    # 研究层
├── long_context/            # Needle-in-a-haystack, 压缩消融
├── longbench_v1/            # LongBench 多任务评测
├── longbench/               # LongBench 风格 QA (v2)
├── loong/                   # 中文多文档 QA
├── zero_scrolls/            # 长文本理解
└── inference_optimization/  # 真实推理测量

pkg/proto/contextengine/     # 共享 gRPC 契约（Go + Python 桩）
scripts/download_benchmark_data.py  # 按需下载 benchmark 数据
```

---

## Context-Engine 设计

context-engine（`services/context-engine/`）在 token 预算内构建优化上下文，分三层：

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│  Retriever  │ → │  Compressor  │ → │   Layout    │
│ BM25/Dense  │   │  tiered      │   │ sink/topic  │
│ keyword     │   │  truncation  │   │             │
└─────────────┘   └──────────────┘   └─────────────┘
```

| 层 | 文件 | 职责 |
|----|------|------|
| strategies | `strategies.py` | 无状态原语：分块、关键词提取、截断、相关度选择、分级压缩、attention-sink 布局 |
| retriever | `retriever.py` | 统一接口 + 工厂：BM25（纯 Python）、关键词、稠密（可选 embedding） |
| pipeline | `pipeline.py` | 编排：检索 → 压缩 → 布局 → 组装 |

引擎以 gRPC 服务暴露（`grpc_server.py`），Go chat-service 通过 `ContextClient`（实现 `domain.ContextOptimizer`）远程调用。

---

## 应用/研究边界

边界明确：

| 层 | 用途 | 数据 | 稳定性 |
|----|------|------|--------|
| `services/` | 生产功能 | 无外部数据集 | 有测试（559+） |
| `research/` | 实验、benchmark、结论 | 大数据集（gitignore） | 探索性 |
| `pkg/proto/` | 共享契约 | - | 稳定接口 |

benchmark 数据集（495MB）不提交到 git，通过 `scripts/download_benchmark_data.py` 按需下载。

---

## 关键发现

实验在 NVIDIA RTX A6000 上运行，模型为 Qwen3-0.6B 和 Qwen2.5-7B。

### LongBench passage_retrieval_en

任务：给定多段落文档，找到与描述匹配的段落。200 样本，每样本约 12.7K tokens。

| 方法 | 准确率 |
|------|--------|
| 截断 | 10% |
| 关键词压缩 | 74% |
| **BM25 检索（top-1）** | **98%** |

BM25 检索命中率 100%（答案段落总在 top-1）。0.6B 模型拿到单个检索段落即可达 98% 准确率。

### 模型尺度不变性

相同压缩上下文，两种模型规模（20 样本）：

| 策略 | Qwen3-0.6B | Qwen2.5-7B |
|------|-----------|------------|
| 截断 | 10% | 10% |
| Project + Topic | 74% | 95% |
| Attention Sink | 60% | 100% |
| Sink + Topic | 60% | 100% |

框架增益跨尺度成立：0.6B 上 7.4 倍、7B 上 10 倍（相对截断）。策略价值随模型能力增长（7B 更能利用布局）。

### 任务边界

| 任务类型 | 框架效果 |
|---------|---------|
| 段落定位（passage_retrieval_en） | 98-100%（杀手锏） |
| 单文档 QA（multifieldqa_en） | F1 0.174 → 0.357（2.1 倍） |
| 科学 QA（qasper） | F1 0.132 → 0.253（1.9 倍） |
| 叙述生成（narrativeqa） | 无增益（需综合生成，非定位） |
| 中文理解/分类 | 有限（0.6B 理解边界） |

---

## 测试覆盖

| 模块 | 测试数 |
|------|--------|
| context-engine | 47（strategies, retriever, pipeline, gRPC） |
| llm-inference | 153 |
| finetune | 115 |
| evaluation | 90 |
| rag | 52 |
| alignment | 50 |
| synthetic-data | 38 |
| rlhf | 21 |
| chat-service (Go) | + context client 测试 |
| long_context 研究 | 14 |

---

## 快速开始

```bash
# 1. 安装依赖
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
pip install torch transformers

# 2. 运行聊天平台
cp .env.example .env
docker compose up -d --build

# 3. 启动 context-engine gRPC 服务
.venv/bin/python -m grpc_server --port 8089

# 4. 运行 benchmark（先下载数据）
python scripts/download_benchmark_data.py
.venv/bin/python research/longbench_v1/run_passage_retrieval.py
```

---

## 仓库大小

benchmark 数据已从 git 排除。仓库约 51MB（源码 + 生成产物），数据按需下载。
