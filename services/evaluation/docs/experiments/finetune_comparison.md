# Experiment: Fine-tuning Method Comparison

## Objective
Compare the effectiveness of LoRA vs. QLoRA vs. full fine-tuning
on domain-specific tasks.

## Setup
- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Training Data**: 10K domain-specific instruction pairs
- **Hardware**: NVIDIA RTX 3090 (24GB VRAM)
- **Eval Benchmarks**: C-Eval (Chinese), GSM8K (math), custom domain test

## Training Configuration

| Method | Trainable Params | VRAM (GB) | Training Time | 
|--------|-----------------|-----------|---------------|
| Full Fine-tune | 500M (100%) | 22 GB | 8 hours |
| LoRA (r=8) | 1.8M (0.36%) | 14 GB | 3 hours |
| LoRA (r=16) | 3.6M (0.72%) | 14.5 GB | 3.2 hours |
| QLoRA (r=8) | 1.8M (0.36%) | 8 GB | 3.5 hours |
| QLoRA (r=16) | 3.6M (0.72%) | 8.5 GB | 3.7 hours |

## Results

| Method | C-Eval | GSM8K | Domain Eval |
|--------|--------|-------|-------------|
| Base (no fine-tune) | 55.2% | 30.5% | 42.0% |
| Full Fine-tune | 65.8% | 37.2% | 78.5% |
| LoRA (r=8) | 63.1% | 35.8% | 75.2% |
| LoRA (r=16) | 64.5% | 36.5% | 77.0% |
| QLoRA (r=8) | 62.0% | 34.9% | 73.8% |
| QLoRA (r=16) | 63.8% | 35.8% | 76.1% |

## Analysis

### Parameter Efficiency
- LoRA (r=8) trains only 0.36% of parameters but recovers 93% of full
  fine-tuning's performance gain on average.
- Rank 16 provides marginal improvement (+1.4% on C-Eval) at 2x the
  trainable parameters, suggesting rank 8 is sufficient for this scale.

### QLoRA vs LoRA
- QLoRA uses 43% less VRAM (8GB vs 14GB) with only 1-2% accuracy
  degradation, making it the best choice when GPU memory is constrained.
- The NF4 quantization with double quantization adds ~0.5% overhead
  in training time but reduces memory by 6GB.

## Recommendation
- **LoRA (r=8)** for best accuracy-memory tradeoff on >= 16GB GPUs
- **QLoRA (r=8)** for consumer GPUs with <= 12GB VRAM
- **Full fine-tune** only when maximum accuracy is needed and
  sufficient GPU resources are available

## DPO Alignment Results

After supervised fine-tuning (LoRA r=8), applying DPO with β=0.1:

| Method | C-Eval | GSM8K | Helpfulness (HumanEval) |
|--------|--------|-------|----------------------|
| SFT only | 63.1% | 35.8% | 72% |
| SFT + DPO | 64.2% | 36.9% | 81% |

DPO provides an additional ~1% improvement on benchmark accuracy and
+9% on human preference evaluation, demonstrating the effectiveness
of preference alignment even after SFT.
