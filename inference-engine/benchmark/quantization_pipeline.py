"""
Quantization benchmark pipeline.

Usage (CI mode, reference data):
    from quantization_pipeline import run_quantization_bench, QuantBenchConfig
    results = run_quantization_bench(QuantBenchConfig())

Usage (GPU required):
    python quantization_pipeline.py --gpu --model Qwen/Qwen2.5-7B-Instruct
"""

# =============================================================================
# WARNING: SIMULATION / ESTIMATION ONLY
#
# The numbers produced by this script are analytical estimates or simulated
# results, NOT real hardware measurements. Do NOT cite these numbers in
# documentation, README, or resumes as measured performance.
#
# Real measured results live in benchmarks/long_context/results/ and were
# obtained by running actual models (Qwen2.5-0.5B, Qwen3-0.6B) on NVIDIA RTX
# A6000 hardware.
# =============================================================================

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class QuantBenchConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    methods: List[str] = field(default_factory=lambda: ["fp16", "int8", "awq", "gptq"])
    benchmarks: List[str] = field(default_factory=lambda: ["mmlu", "gsm8k"])
    batch_size: int = 1
    max_tokens: int = 128
    use_gpu: bool = False  # set True on GPU hardware


# Reference data for Qwen2.5-7B
_REFERENCE = {
    "fp16": {"vram_gb": 14.0, "latency_ms": 45.0, "mmlu": 0.701, "gsm8k": 0.523},
    "int8": {"vram_gb": 8.5, "latency_ms": 38.0, "mmlu": 0.698, "gsm8k": 0.518},
    "awq": {"vram_gb": 5.0, "latency_ms": 32.0, "mmlu": 0.695, "gsm8k": 0.515},
    "gptq": {"vram_gb": 5.5, "latency_ms": 35.0, "mmlu": 0.688, "gsm8k": 0.505},
    "fp8": {"vram_gb": 7.0, "latency_ms": 34.0, "mmlu": 0.700, "gsm8k": 0.520},
}


def compute_memory_savings(method: str) -> float:
    """Compute memory savings ratio relative to FP16."""
    ref = _REFERENCE.get(method)
    fp16 = _REFERENCE.get("fp16", {"vram_gb": 14.0})
    if not ref:
        return 0.0
    return 1.0 - ref["vram_gb"] / fp16["vram_gb"]


def compute_accuracy_impact(method: str) -> float:
    """Compute accuracy degradation relative to FP16."""
    ref = _REFERENCE.get(method)
    fp16 = _REFERENCE.get("fp16", {"mmlu": 0.7})
    if not ref:
        return 0.0
    return fp16["mmlu"] - ref["mmlu"]


def run_quantization_bench(config: QuantBenchConfig) -> List[Dict]:
    """Run quantization benchmark.

    On GPU: loads model with each method and measures actual performance.
    In CI: returns reference data.
    """
    if config.use_gpu:
        return _run_on_gpu(config)
    return _run_reference(config)


def _run_reference(config: QuantBenchConfig) -> List[Dict]:
    results = []
    for method in config.methods:
        ref = _REFERENCE.get(method)
        if ref is None:
            continue
        results.append({
            "method": method.upper(),
            "model": config.model_name,
            **ref,
        })
    return results


def _run_on_gpu(config: QuantBenchConfig) -> List[Dict]:
    """Run actual model loading and inference on GPU hardware."""
    _src = os.path.join(os.path.dirname(__file__), "..", "..", "services", "llm-inference", "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    results = []
    for method in config.methods:
        quant = None if method == "fp16" else method
        try:
            from engine_factory import EngineFactory, EngineType
            engine = EngineFactory.create(
                engine_type=EngineType.AUTO,
                model_path=config.model_name,
                quantization=quant,
                max_tokens=config.max_tokens,
            )
            # Measure memory
            import torch
            vram = torch.cuda.memory_allocated() / (1024**3)

            # Measure latency
            import time
            start = time.time()
            response = engine.generate([{"role": "user", "content": "Benchmark test."}])
            elapsed = (time.time() - start) * 1000
            latency_ms = elapsed / max(response.generated_tokens, 1)

            results.append({
                "method": method.upper(),
                "model": config.model_name,
                "vram_gb": round(vram, 1),
                "latency_ms": round(latency_ms, 1),
                "mmlu": 0.0,  # Would require running actual evaluation
                "gsm8k": 0.0,
            })
            engine.close()
        except Exception as e:
            results.append({
                "method": method.upper(),
                "model": config.model_name,
                "error": str(e),
            })
    return results


def format_results(results: List[Dict]) -> str:
    """Format results as a markdown table."""
    lines = [
        "| Method | VRAM (GB) | Latency (ms/t) | MMLU | GSM8K |",
        "|--------|-----------|----------------|------|-------|",
    ]
    for r in results:
        mmlu = f"{r['mmlu']:.1%}" if r.get('mmlu', 0) > 0 else "-"
        gsm8k = f"{r['gsm8k']:.1%}" if r.get('gsm8k', 0) > 0 else "-"
        latency = f"{r['latency_ms']}" if r.get('latency_ms') else "-"
        vram = f"{r['vram_gb']}" if r.get('vram_gb') else "-"
        lines.append(f"| {r['method']} | {vram} | {latency} | {mmlu} | {gsm8k} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Run on real GPU hardware")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--methods", nargs="+", default=["fp16", "int8", "awq", "gptq"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = QuantBenchConfig(
        model_name=args.model,
        methods=args.methods,
        use_gpu=args.gpu,
    )

    print(f"Quantization benchmark: {config.model_name}")
    print(f"Methods: {', '.join(m.upper() for m in config.methods)}")
    print(f"Mode: {'GPU' if args.gpu else 'reference (CI)'}")
    print()

    results = run_quantization_bench(config)
    print(format_results(results))
    print()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
