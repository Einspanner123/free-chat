"""
真实推理优化收益测量（RTX A6000 + 真实模型）。

测量：
1. Prefill 延迟 vs 输入长度 —— 验证上下文越长 prefill 越慢（压缩的价值）
2. Decode 延迟（TPOT）vs 上下文长度 —— 验证长上下文下每 token 生成成本
3. Prefix Cache 跨请求复用 —— 相同前缀第二次请求的 prefill 时间节省
4. 压缩 vs 全量上下文的总延迟对比 —— 端到端收益

用法：
    .venv/bin/python benchmarks/inference_optimization/run_real.py
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# 真实书籍文本
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "long_context", "data")


def load_book_text(name: str = "pride_and_prejudice") -> str:
    with open(os.path.join(DATA_DIR, f"{name}.txt"), "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def measure_prefill(model, tok, device, input_ids, num_runs: int = 3) -> float:
    """测量 prefill（prompt 编码）延迟：只做一次 forward pass 不生成。"""
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.time()
            model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    return sum(times) / len(times)


def measure_decode(model, tok, device, input_ids, max_new: int = 32) -> Dict:
    """测量 decode 延迟：生成 max_new 个 token，计算 TPOT。"""
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) * 1000
    gen_tokens = max_new
    tpot = elapsed / gen_tokens
    return {"total_ms": elapsed, "gen_tokens": gen_tokens, "tpot_ms": tpot}


def measure_prefix_cache(model, tok, device, prefix_ids, suffix_ids, num_runs: int = 3) -> Dict:
    """
    测量 Prefix Cache 收益：
    - 第一次：完整 prefill（prefix + suffix）
    - 第二次：复用 prefix 的 KV cache，只 prefill suffix
    
    用真实 past_key_values 缓存实现。
    """
    # 第一次：完整 prefill，缓存 prefix 的 KV
    full_input = torch.cat([prefix_ids, suffix_ids], dim=1)

    with torch.no_grad():
        # 完整 prefill 时间
        t0 = time.time()
        out = model(input_ids=full_input, use_cache=True)
        torch.cuda.synchronize()
        full_prefill_ms = (time.time() - t0) * 1000

        # 提取 prefix 的 KV cache（前 prefix_len 个位置），用 DynamicCache
        prefix_len = prefix_ids.shape[1]
        cache = DynamicCache()
        if hasattr(out, 'past_key_values') and out.past_key_values is not None:
            pkv = out.past_key_values
            keys = pkv.key_cache if hasattr(pkv, 'key_cache') else [kv[0] for kv in pkv]
            values = pkv.value_cache if hasattr(pkv, 'value_cache') else [kv[1] for kv in pkv]
            for i, (k, v) in enumerate(zip(keys, values)):
                cache.update(k[:, :, :prefix_len, :], v[:, :, :prefix_len, :], layer_idx=i)
        past_kv = cache

    # 第二次：只用 suffix，附加缓存的 prefix KV
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.time()
            model(input_ids=suffix_ids, past_key_values=past_kv, use_cache=True)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    cached_prefill_ms = sum(times) / len(times)

    return {
        "prefix_tokens": prefix_len,
        "suffix_tokens": suffix_ids.shape[1],
        "full_prefill_ms": round(full_prefill_ms, 1),
        "cached_prefill_ms": round(cached_prefill_ms, 1),
        "prefill_saved_ms": round(full_prefill_ms - cached_prefill_ms, 1),
        "speedup": round(full_prefill_ms / cached_prefill_ms, 2) if cached_prefill_ms > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--output", default="results/inference_opt_real.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    book = load_book_text(args.book)

    results = {"config": {"model": args.model, "book": args.book}, "measurements": {}}

    # ============================================================
    # 1. Prefill 延迟 vs 输入长度
    # ============================================================
    print("\n=== 1. Prefill latency vs input length ===")
    prefill_rows = []
    for target_tokens in [256, 512, 1024, 2048, 4096]:
        input_text = tok.decode(tok.encode(book, add_special_tokens=False)[:target_tokens], skip_special_tokens=True)
        ids = tok(input_text, return_tensors="pt").input_ids.to(device)
        ms = measure_prefill(model, tok, device, ids)
        prefill_rows.append({"input_tokens": ids.shape[1], "prefill_ms": round(ms, 1)})
        print(f"  {ids.shape[1]:>5} tokens: {ms:>8.1f} ms")
    results["measurements"]["prefill_scaling"] = prefill_rows

    # ============================================================
    # 2. Decode TPOT vs 上下文长度
    # ============================================================
    print("\n=== 2. Decode TPOT vs context length ===")
    decode_rows = []
    for target_tokens in [256, 1024, 4096]:
        input_text = tok.decode(tok.encode(book, add_special_tokens=False)[:target_tokens], skip_special_tokens=True)
        ids = tok(input_text, return_tensors="pt").input_ids.to(device)
        d = measure_decode(model, tok, device, ids, max_new=32)
        decode_rows.append({"context_tokens": ids.shape[1], "tpot_ms": round(d["tpot_ms"], 2), "gen_ms": round(d["total_ms"], 1)})
        print(f"  context={ids.shape[1]:>5} tokens: TPOT={d['tpot_ms']:.2f}ms/token, gen 32 tok = {d['total_ms']:.0f}ms")
    results["measurements"]["decode_scaling"] = decode_rows

    # ============================================================
    # 3. Prefix Cache 跨请求复用
    # ============================================================
    print("\n=== 3. Prefix cache cross-request reuse ===")
    # 用真实书籍前 2048 tokens 作为共享前缀，两个不同的后缀问题
    prefix_text = tok.decode(tok.encode(book, add_special_tokens=False)[:2048], skip_special_tokens=True)
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids.to(device)
    suffix_a = tok(" What is the capital of France?", return_tensors="pt").input_ids.to(device)
    suffix_b = tok(" Who wrote the book?", return_tensors="pt").input_ids.to(device)

    for suffix_name, suffix in [("suffix_A", suffix_a), ("suffix_B", suffix_b)]:
        m = measure_prefix_cache(model, tok, device, prefix_ids, suffix)
        print(f"  {suffix_name}: prefix={m['prefix_tokens']} tok, full prefill={m['full_prefill_ms']}ms, "
              f"cached prefill={m['cached_prefill_ms']}ms, saved={m['prefill_saved_ms']}ms ({m['speedup']}x)")
        results["measurements"][f"prefix_cache_{suffix_name}"] = m

    # ============================================================
    # 4. 压缩 vs 全量的端到端延迟
    # ============================================================
    print("\n=== 4. End-to-end: full context vs compressed (88%) ===")
    full_input = tok.decode(tok.encode(book, add_special_tokens=False)[:4096], skip_special_tokens=True)
    compressed = full_input[:512]  # 模拟 88% 压缩
    question = " What is the main theme?"
    full_prompt = full_input + question
    comp_prompt = compressed + question

    full_ids = tok(full_prompt, return_tensors="pt").input_ids.to(device)
    comp_ids = tok(comp_prompt, return_tensors="pt").input_ids.to(device)

    full_prefill = measure_prefill(model, tok, device, full_ids)
    comp_prefill = measure_prefill(model, tok, device, comp_ids)
    full_decode = measure_decode(model, tok, device, full_ids, max_new=32)
    comp_decode = measure_decode(model, tok, device, comp_ids, max_new=32)

    full_total = full_prefill + full_decode["total_ms"]
    comp_total = comp_prefill + comp_decode["total_ms"]
    print(f"  Full  ({full_ids.shape[1]:>4} tok): prefill={full_prefill:.0f}ms + decode={full_decode['total_ms']:.0f}ms = {full_total:.0f}ms")
    print(f"  Comp  ({comp_ids.shape[1]:>4} tok): prefill={comp_prefill:.0f}ms + decode={comp_decode['total_ms']:.0f}ms = {comp_total:.0f}ms")
    print(f"  End-to-end speedup: {full_total/comp_total:.2f}x")
    results["measurements"]["e2e_compression"] = {
        "full_tokens": full_ids.shape[1], "compressed_tokens": comp_ids.shape[1],
        "full_total_ms": round(full_total, 1), "compressed_total_ms": round(comp_total, 1),
        "speedup": round(full_total / comp_total, 2),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
