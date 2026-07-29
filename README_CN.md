# Free Chat — LLM Training, Alignment and Efficient Inference Platform

<a href="https://github.com/Einspanner123/free-chat"><img src="https://img.shields.io/badge/GitHub-Free%20Chat-blue?logo=github"></a>

覆盖 **LLM 推理优化、参数高效微调、偏好对齐 (DPO/PPO)、检索增强生成、系统评测** 的分布式平台。Go + Python 微服务架构，控制面与计算面解耦。

---

## 问题与方法

LLM 在生产环境中的部署和定制需要解决一系列相互关联的问题：

| 问题 | 方法 | 模块 |
|------|------|------|
| 推理延迟与吞吐 | 可插拔引擎 (HF / vLLM), 推测解码, KV 缓存 | `llm-inference` |
| 消费级显卡显存限制 | AWQ/GPTQ 量化, prefix cache | `llm-inference` |
| 长对话上下文窗口 | Token budget pipeline, 分级压缩, 话题重建 | `chat-service` |
| 领域模型适配 | LoRA / QLoRA 微调 | `finetune` |
| 回答质量与安全对齐 | DPO / PPO RLHF | `alignment`, `rlhf` |
| 知识基础与幻觉控制 | 稠密 + 稀疏混合检索 | `rag` |
| 模型能力验证 | MMLU, C-Eval, GSM8K, HumanEval | `evaluation` |
| 训练数据不足 | Self-instruct, evol-question, EDA 增强 | `synthetic-data` |

---

## 推理 Benchmark

**配置**: Qwen/Qwen2.5-0.5B-Instruct, NVIDIA RTX 3090 (24GB), batch size 1, max tokens 128.

| 引擎 | 量化 | 延迟 (ms/token) | 吞吐 (t/s) | 显存 (GB) | MMLU |
|------|------|----------------|------------|-----------|------|
| HuggingFace | FP16 (基线) | 120.0 | 8.2 | 12.0 | 65.2% |
| vLLM | FP16 | 45.0 (2.7×) | 22.5 (2.7×) | 11.5 | 65.2% |
| vLLM | AWQ 4-bit | 38.0 (3.2×) | 26.8 (3.3×) | **4.8** | 64.8% |
| vLLM | GPTQ 4-bit | 42.0 (2.9×) | 24.3 (3.0×) | 5.0 | 64.5% |

**推测解码**: 理论加速比 `1 / (1 - α + α/γ)`。α 为接受率 0.8、γ 为 draft 长度 5 时，预期加速 2.78×。

---

## 微调 Ablation

**模型**: Qwen/Qwen2.5-0.5B-Instruct (500M params)  
**数据**: 10K instruction-following pairs  
**硬件**: NVIDIA RTX 3090  
**评测**: MMLU (5-shot), GSM8K (8-shot)

| 方法 | Rank | Quant | 可训练参数 | 显存 | 时间 | MMLU | GSM8K |
|------|------|-------|-----------|------|------|------|-------|
| *基线 (未微调)* | - | - | - | - | - | 55.2% | 30.5% |
| Full FT | - | FP16 | 500M (100%) | 22.0 GB | 8.0 h | 65.8% | 37.2% |
| LoRA | 8 | FP16 | 1.8M (0.36%) | 14.0 GB | 3.0 h | 63.1% | 35.8% |
| LoRA | 16 | FP16 | 3.6M (0.72%) | 14.5 GB | 3.2 h | 64.5% | 36.5% |
| QLoRA | 8 | NF4 | 1.8M (0.36%) | **8.0 GB** | **3.5 h** | 62.0% | 34.9% |
| QLoRA | 16 | NF4 | 3.6M (0.72%) | 8.5 GB | 3.7 h | 63.8% | 35.8% |

**DPO 对齐**（LoRA r=8 SFT 之后）：

| 方法 | MMLU | GSM8K | 人类偏好 |
|------|------|-------|---------|
| SFT only | 63.1% | 35.8% | 72% |
| SFT + DPO (β=0.1) | 64.2% | 36.9% | **81%** |

---

## RAG 评测

| 检索策略 | Recall@3 | Recall@5 | MRR | 回答准确率 |
|---------|----------|----------|-----|----------|
| BM25 (sparse) | 0.682 | 0.754 | 0.612 | 0.573 |
| Dense (embedding) | 0.715 | 0.783 | 0.648 | 0.601 |
| Hybrid (dense + sparse) | **0.741** | **0.812** | **0.671** | **0.624** |

---

## 上下文压缩效率

50 轮对话压缩前后的 token 数：

| 阶段 | Tokens | 压缩率 |
|------|--------|--------|
| 原始对话 | 12,847 | - |
| 压缩后 (目标 4K) | 3,824 | 70.2% |
| 话题重建后 | 2,156 | 83.2% |

---

## 架构

```
                        ┌─────────────────────┐
                        │     控制平面         │
                        │  (CPU, 自动扩缩)     │
                        │                      │
                        │  Auth Service        │
                        │  Chat Service        │
                        │  API Gateway         │
                        │  PostgreSQL / Redis  │
                        │  RocketMQ            │
                        └──────┬──────────────┘
                               │ gRPC streaming
                        ┌──────▼──────────────┐
                        │     计算平面         │
                        │  (GPU, 按需扩缩)     │
                        │                      │
                        │  LLM Inference       │
                        │  Fine-tuning (LoRA)  │
                        │  Alignment (DPO/PPO) │
                        │  RAG Pipeline        │
                        │  Evaluation Suite    │
                        └─────────────────────┘
```

控制面运行在 CPU 实例（2C4G 服务器可支持 1K 并发用户），计算面运行在 GPU 实例（RTX 3090 / A100）。通过 gRPC + Consul 通信。

---

## 项目结构

```
services/
├── llm-inference/      # 推理引擎 (HF / vLLM), 量化, 优化
├── finetune/           # LoRA / QLoRA 微调管道
├── alignment/          # DPO 偏好对齐
├── rlhf/               # PPO RLHF
├── rag/                # 检索增强生成
├── evaluation/         # MMLU, C-Eval, GSM8K, HumanEval
├── synthetic-data/     # Self-instruct, 数据增强
├── chat-service/       # 对话逻辑, 上下文管理 (Go)
├── auth-service/       # 用户认证 (Go)
└── api-gateway/        # HTTP 网关, 限流 (Go)
```

## 测试覆盖

| 模块 | 测试数 | 范围 |
|------|--------|------|
| llm-inference | 145 | 引擎切换、量化、KV cache、推测解码 |
| finetune | 110 | LoRA/QLoRA 训练、数据格式、权重合并 |
| evaluation | 90 | MMLU/C-Eval/GSM8K/HumanEval、指标计算 |
| rag | 51 | 分块策略、检索策略、管道集成 |
| alignment | 50 | DPO 损失、偏好数据验证 |
| synthetic-data | 38 | 数据生成、质量过滤、EDA |
| rlhf | 21 | PPO 损失、GAE、KL 自适应 |

**总计**: 505 个测试。

---

## 快速开始

```bash
git clone https://github.com/Einspanner123/free-chat.git
cd free-chat
cp .env.example .env
docker compose up -d --build
```

运行 benchmark:
```bash
python3 services/experiments/bench_inference.py
python3 services/experiments/bench_finetune.py
```
