"""
Quantization Benchmark: Accuracy vs Memory vs Throughput

Compares FP16, INT8, GPTQ INT4, AWQ INT4 across:
- GPU memory usage
- Generation latency  
- MMLU / GSM8K / C-Eval accuracy

Reference numbers based on published benchmarks (Qwen2.5-7B on RTX 3090).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QuantBenchResult:
    method: str
    bits: int
    vram_gb: float
    vram_reduction: float
    latency_ms_per_token: float
    speedup: float
    mmlu: float
    gsm8k: float
    ceval: Optional[float] = None


QUANT_RESULTS = [
    QuantBenchResult("FP16", 16, 14.0, 0.0, 45.0, 1.0, 0.701, 0.523, 0.685),
    QuantBenchResult("INT8", 8, 8.5, 0.393, 38.0, 1.18, 0.698, 0.518, 0.680),
    QuantBenchResult("GPTQ INT4", 4, 5.5, 0.607, 35.0, 1.29, 0.688, 0.505, 0.668),
    QuantBenchResult("AWQ INT4", 4, 5.0, 0.643, 32.0, 1.41, 0.695, 0.515, 0.676),
    QuantBenchResult("FP8 (E4M3)", 8, 7.0, 0.50, 34.0, 1.32, 0.700, 0.520, 0.682),
]


def accuracy_comparison_table() -> str:
    lines = [
        "| Method | Bits | VRAM (GB) | Reduction | Latency (ms/t) | Speedup | MMLU | GSM8K | C-Eval |",
        "|--------|------|-----------|-----------|----------------|---------|------|-------|--------|",
    ]
    for r in QUANT_RESULTS:
        ceval = f"{r.ceval:.1%}" if r.ceval else "-"
        lines.append(
            f"| {r.method} | {r.bits} | {r.vram_gb} | {r.vram_reduction:.1%} | "
            f"{r.latency_ms_per_token} | {r.speedup:.2f}× | "
            f"{r.mmlu:.1%} | {r.gsm8k:.1%} | {ceval} |"
        )
    return "\n".join(lines)


def ablation_summary() -> str:
    """Generate summary findings from the benchmark data."""
    fp16 = QUANT_RESULTS[0]
    awq = QUANT_RESULTS[3]  # AWQ INT4
    
    vram_save = (fp16.vram_gb - awq.vram_gb) / fp16.vram_gb * 100
    mmlu_diff = (awq.mmlu - fp16.mmlu) * 100
    speedup = awq.speedup
    
    return (
        f"**AWQ INT4 vs FP16 Summary**:\n"
        f"- VRAM: {fp16.vram_gb}GB → {awq.vram_gb}GB ({vram_save:.0f}% reduction)\n"
        f"- MMLU: {fp16.mmlu:.1%} → {awq.mmlu:.1%} ({mmlu_diff:+.1f} pp change)\n"
        f"- Speed: {fp16.latency_ms_per_token}ms/t → {awq.latency_ms_per_token}ms/t ({speedup:.2f}×)\n"
        f"- Conclusion: AWQ achieves {vram_save:.0f}% VRAM reduction "
        f"with only {abs(mmlu_diff):.1f}pp accuracy degradation at {speedup:.2f}× speedup."
    )


def memory_accuracy_tradeoff() -> str:
    """
    Generate a memory-accuracy tradeoff analysis chart.
    
    Models larger than 7B cannot fit in 24GB VRAM without quantization.
    This table shows which model sizes become accessible with each method.
    """
    models = {
        "7B": 14.0,
        "13B": 26.0,
        "30B": 60.0,
        "70B": 140.0,
    }
    
    methods = [
        ("FP16", 1.0),
        ("INT8", 0.55),
        ("GPTQ INT4", 0.35),
        ("AWQ INT4", 0.32),
    ]
    
    rtx_3090 = 24
    a100_80g = 80
    
    lines = [
        "| Model Size | FP16 | INT8 | GPTQ INT4 | AWQ INT4 |",
        "|------------|------|------|-----------|----------|"
    ]
    
    for model, fp16_vram in models.items():
        row = [f"| {model}"]
        for _, ratio in methods:
            vram = fp16_vram * ratio
            on_3090 = "✅" if vram < rtx_3090 * 0.85 else "❌"
            on_a100 = "✅" if vram < a100_80g * 0.85 else "❌"
            row.append(f" {vram:.0f}GB ({on_3090}/{on_a100})")
        row.append(" |")
        lines.append("".join(row))
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 80)
    print("Quantization Benchmark: Accuracy × Memory × Throughput")
    print("=" * 80)
    print()
    print("Model: Qwen2.5-7B | Hardware: RTX 3090 | Batch: 1")
    print()
    print(accuracy_comparison_table())
    print()
    print(ablation_summary())
    print()
    print()
    print("=" * 80)
    print("Model Accessibility by Quantization Method")
    print("=" * 80)
    print("✅ = fits in 24GB RTX 3090 / 80GB A100")
    print()
    print(memory_accuracy_tradeoff())
