"""
真实 decode 优化测量（RTX A6000 + Qwen3-0.6B）。

1. 批量 decode：batch_size 1/2/4/8，验证 memory-bound 下带宽摊薄
2. 量化 decode：FP16 vs INT8（bitsandbytes 8bit），验证权重字节减少的加速

用法：
    .venv/bin/python benchmarks/inference_optimization/run_decode_optimization.py
"""

import argparse
import json
import os
import statistics
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "long_context", "data")


def load_book_text(name: str = "pride_and_prejudice") -> str:
    with open(os.path.join(DATA_DIR, f"{name}.txt"), "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def median_time(fn, runs: int = 5) -> float:
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    return statistics.median(times)


def measure_batch_decode(model, tok, device, batch_size: int, context_ids, max_new: int = 32, runs: int = 3) -> Dict:
    """
    批量 decode：batch_size 个请求一起生成。
    decode 是 memory-bound，批量共享权重读取，带宽摊薄。
    """
    # 构造 batch：每个请求不同上下文（但同样长度）
    batch_inputs = context_ids.repeat(batch_size, 1)

    def decode_batch():
        model.generate(
            input_ids=batch_inputs,
            max_new_tokens=max_new,
            do_sample=False,
        )

    total_ms = median_time(decode_batch, runs)
    per_batch_tokens = batch_size * max_new
    throughput = per_batch_tokens / (total_ms / 1000)

    return {
        "batch_size": batch_size,
        "total_ms_for_batch": round(total_ms, 1),
        "tokens_per_second": round(throughput, 1),
        "per_request_ms": round(total_ms / batch_size, 1),
    }


def measure_quantized_decode(model, tok, device, context_ids, max_new: int = 32, runs: int = 3) -> Dict:
    """测量量化模型的 decode 延迟。"""
    def decode():
        model.generate(
            input_ids=context_ids,
            max_new_tokens=max_new,
            do_sample=False,
        )
    total_ms = median_time(decode, runs)
    return {
        "max_new": max_new,
        "total_ms": round(total_ms, 1),
        "tpot_ms": round(total_ms / max_new, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", default="results/decode_optimization.json")
    args = parser.parse_args()

    device = "cuda"
    results = {"config": {"model": args.model, "book": args.book}, "measurements": {}}

    # ============================================================
    # 1. 批量 decode
    # ============================================================
    print("Loading FP16 model...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    book = load_book_text(args.book)
    ctx = tok.decode(tok.encode(book, add_special_tokens=False)[:2048], skip_special_tokens=True)
    context_ids = tok(ctx, return_tensors="pt").input_ids.to(device)

    print("\n=== 1. Batch decode (FP16, 2048-token context, 32 gen tokens) ===\n")
    batch_rows = []
    for bs in args.batch_sizes:
        r = measure_batch_decode(model_fp16, tok, device, bs, context_ids)
        batch_rows.append(r)
        print(f"  batch={bs}: {r['tokens_per_second']:>8.0f} tokens/s total, {r['per_request_ms']:.0f}ms/request")
    results["measurements"]["batch_decode"] = batch_rows

    # ============================================================
    # 2. 量化 decode (FP16 vs INT8)
    # ============================================================
    print("\n=== 2. Quantized decode: FP16 vs INT8 ===\n")
    fp16_row = measure_quantized_decode(model_fp16, tok, device, context_ids, max_new=32)
    print(f"  FP16: {fp16_row['total_ms']:.0f}ms for 32 tokens (TPOT={fp16_row['tpot_ms']}ms)")
    del model_fp16
    torch.cuda.empty_cache()

    int8_row = None
    try:
        import bitsandbytes as bnb
        print("Loading INT8 (bitsandbytes) model...")
        model_int8 = AutoModelForCausalLM.from_pretrained(
            args.model, load_in_8bit=True, trust_remote_code=True,
            device_map="cuda",
        )
        int8_row = measure_quantized_decode(model_int8, tok, device, context_ids, max_new=32)
        print(f"  INT8: {int8_row['total_ms']:.0f}ms for 32 tokens (TPOT={int8_row['tpot_ms']}ms)")
        del model_int8
        torch.cuda.empty_cache()
    except ImportError as e:
        print(f"  bitsandbytes not available: {e}")
    except Exception as e:
        print(f"  INT8 load failed: {e}")

    results["measurements"]["quantized_decode"] = {
        "fp16": fp16_row,
        "int8": int8_row,
    }

    if int8_row:
        speedup = fp16_row["tpot_ms"] / int8_row["tpot_ms"]
        print(f"\n  INT8 decode speedup vs FP16: {speedup:.2f}x")
        results["measurements"]["quantized_decode"]["int8_speedup"] = round(speedup, 2)

    # ============================================================
    # 3. 批量 + 吞吐分析
    # ============================================================
    print("\n=== 3. Throughput scaling analysis ===\n")
    if len(batch_rows) >= 2:
        base = batch_rows[0]["tokens_per_second"]
        for r in batch_rows:
            ratio = r["tokens_per_second"] / base if base > 0 else 0
            print(f"  batch={r['batch_size']}: {ratio:.2f}x throughput vs batch=1")
            r["throughput_vs_batch1"] = round(ratio, 2)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
