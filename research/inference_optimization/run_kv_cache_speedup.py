"""
真实 KV Cache 推理加速测量（RTX A6000 + Qwen3-0.6B）。

测量两个真实可复现的推理加速：
1. Prefix Cache 跨请求复用：共享前缀跳过 prefill
   （vLLM Prefix Caching 同思路，真实 past_key_values 复用）
2. KV Cache 截断对 decode 的加速：减少 attention 计算量
   （Sliding Window Eviction 的真实效果）

每个测量多次运行取中位数，保证稳定。
"""

import argparse
import json
import os
import statistics
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "long_context", "data")


def load_book_text(name: str = "pride_and_prejudice") -> str:
    with open(os.path.join(DATA_DIR, f"{name}.txt"), "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def median_time(fn, runs: int = 5) -> float:
    """运行 fn 多次取中位数（毫秒）。"""
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    return statistics.median(times)


def measure_prefix_cache(model, tok, device, prefix_ids, suffix_ids, runs: int = 5) -> Dict:
    """Prefix Cache：完整 prefill vs 复用 prefix KV + 只 prefill suffix。"""
    full_input = torch.cat([prefix_ids, suffix_ids], dim=1)

    # 完整 prefill（不含复用）
    def full_prefill():
        model(input_ids=full_input, use_cache=True)
    full_ms = median_time(full_prefill, runs)

    # 提取 prefix KV cache
    with torch.no_grad():
        out = model(input_ids=full_input, use_cache=True)
    prefix_len = prefix_ids.shape[1]
    cache = DynamicCache()
    keys = out.past_key_values.key_cache if hasattr(out.past_key_values, "key_cache") else [kv[0] for kv in out.past_key_values]
    values = out.past_key_values.value_cache if hasattr(out.past_key_values, "value_cache") else [kv[1] for kv in out.past_key_values]
    for i, (k, v) in enumerate(zip(keys, values)):
        cache.update(k[:, :, :prefix_len, :], v[:, :, :prefix_len, :], layer_idx=i)

    # 复用 prefix KV + 只 prefill suffix
    def cached_prefill():
        model(input_ids=suffix_ids, past_key_values=cache, use_cache=True)
    cached_ms = median_time(cached_prefill, runs)

    return {
        "prefix_tokens": prefix_len,
        "suffix_tokens": suffix_ids.shape[1],
        "full_prefill_ms": round(full_ms, 1),
        "cached_prefill_ms": round(cached_ms, 1),
        "saved_ms": round(full_ms - cached_ms, 1),
        "speedup": round(full_ms / cached_ms, 2) if cached_ms > 0 else 0,
    }


def manual_decode(model, tok, device, cache, first_token_id, max_new: int = 32):
    """手动 decode 循环：从给定 KV cache 开始，迭代生成 max_new 个 token。"""
    input_ids = first_token_id
    for _ in range(max_new):
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cache = out.past_key_values
        input_ids = next_token


def make_cache(keys, values, keep: int = None, layer_idx_start: int = 0):
    """构造 DynamicCache，可选截断到最近 keep 个 token。"""
    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(keys, values)):
        if keep is not None:
            cache.update(k[:, :, -keep:, :], v[:, :, -keep:, :], layer_idx=i + layer_idx_start)
        else:
            cache.update(k, v, layer_idx=i + layer_idx_start)
    return cache


def measure_kv_eviction_decode(model, tok, device, context_ids, keep_tokens: List[int], max_new: int = 32, runs: int = 3) -> Dict:
    """
    KV Cache 截断对 decode 的加速（手动 decode 循环）。
    把长上下文的 KV cache 截断到 keep_tokens，decode 时 attention 计算量减少。
    """
    with torch.no_grad():
        out = model(input_ids=context_ids, use_cache=True)
    context_len = context_ids.shape[1]
    keys = out.past_key_values.key_cache if hasattr(out.past_key_values, "key_cache") else [kv[0] for kv in out.past_key_values]
    values = out.past_key_values.value_cache if hasattr(out.past_key_values, "value_cache") else [kv[1] for kv in out.past_key_values]

    # decode 起点：完整 KV cache（含全部 context），输入为最后一个 token
    first_token = context_ids[:, -1:]

    results = []

    # Baseline: 完整 KV cache decode（用除最后一个外的全部，避免重复）
    def full_decode():
        cache = make_cache(keys, values)
        manual_decode(model, tok, device, cache, first_token, max_new)
    full_ms = median_time(full_decode, runs)

    for keep in keep_tokens:
        if keep >= context_len:
            continue
        def evicted_decode():
            cache = make_cache(keys, values, keep=keep)
            manual_decode(model, tok, device, cache, first_token, max_new)
        evicted_ms = median_time(evicted_decode, runs)
        results.append({
            "context_tokens": context_len,
            "keep_tokens": keep,
            "eviction_ratio": round(1 - keep / context_len, 3),
            "full_decode_ms": round(full_ms, 1),
            "evicted_decode_ms": round(evicted_ms, 1),
            "saved_ms": round(full_ms - evicted_ms, 1),
            "speedup": round(full_ms / evicted_ms, 2) if evicted_ms > 0 else 0,
        })

    return {"baseline_full_ms": round(full_ms, 1), "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--output", default="results/kv_cache_speedup.json")
    args = parser.parse_args()

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    book = load_book_text(args.book)

    results = {"config": {"model": args.model, "book": args.book}, "measurements": {}}

    # ============================================================
    # 1. Prefix Cache 跨请求复用
    # ============================================================
    print("=== 1. Prefix Cache: shared prefix across requests ===\n")
    prefix_text = tok.decode(tok.encode(book, add_special_tokens=False)[:4096], skip_special_tokens=True)
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids.to(device)
    suffixes = {
        "short_suffix": " What is the capital?",
        "long_suffix": " Given the context above, what is the main theme and how does it develop over the course of the narrative?",
    }
    for name, s in suffixes.items():
        suffix_ids = tok(s, return_tensors="pt").input_ids.to(device)
        m = measure_prefix_cache(model, tok, device, prefix_ids, suffix_ids)
        print(f"  {name}: prefix={m['prefix_tokens']} tok, full={m['full_prefill_ms']}ms, "
              f"cached={m['cached_prefill_ms']}ms, saved={m['saved_ms']}ms ({m['speedup']}x)")
        results["measurements"][f"prefix_cache_{name}"] = m

    # ============================================================
    # 2. KV Cache 截断（Sliding Window）对 decode 的加速
    # ============================================================
    print("\n=== 2. KV cache eviction: decode speedup ===\n")
    context_text = tok.decode(tok.encode(book, add_special_tokens=False)[:4096], skip_special_tokens=True)
    context_ids = tok(context_text, return_tensors="pt").input_ids.to(device)
    keep_list = [1024, 2048, 3072]
    eviction = measure_kv_eviction_decode(model, tok, device, context_ids, keep_list, max_new=32, runs=3)
    print(f"  Baseline (full {context_ids.shape[1]} tok KV): {eviction['baseline_full_ms']}ms for 32 tokens")
    for r in eviction["results"]:
        print(f"  Keep {r['keep_tokens']:>4} tok ({r['eviction_ratio']:.0%} evicted): {r['evicted_decode_ms']}ms "
              f"saved={r['saved_ms']}ms ({r['speedup']}x)")
    results["measurements"]["kv_eviction_decode"] = eviction

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
