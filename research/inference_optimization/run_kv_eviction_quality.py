"""KV 逐出(StreamingLLM 式)的 显存 × 质量 基准 — 真实硬件。

背景:现有 `kv_cache_speedup.json` 的 `kv_eviction_decode` 只测了解码速度
(0.97–1.0x,逐出不加速)。但 KV 逐出的价值是**显存**:同等显存下上下文/batch
翻倍或换更小卡。本基准补齐两个被遗漏的指标:

1. 显存:prefill 后 decode 的峰值显存 + KV 字节(解析精确),逐出率/上下文倍率
2. 质量:NIAH recall-by-position(prefill 全量,decode 逐出后检索)+ 可选 perplexity

实现:直接注入 `SinkWindowCache`(attention-sink + sliding window,transformers
Cache 子类,见 services/llm-inference/src/optimization/kv_eviction.py)到
`model.generate(past_key_values=...)` 真实路径。

用法:
    .venv/bin/python research/inference_optimization/run_kv_eviction_quality.py \
        --context-tokens 16384 --windows 256 512 1024 2048 --with-perplexity
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ---- bootstrap: import the service-layer algorithm + long_context harness ----
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "services" / "llm-inference" / "src"))
sys.path.insert(0, str(REPO / "research" / "long_context"))

from optimization.kv_eviction import SinkWindowCache, kv_cache_bytes  # noqa: E402
from metrics import compute_position_recall  # noqa: E402
from run_realtext import (  # noqa: E402
    insert_needles_mixed,
    load_real_text,
    prepare_real_context,
)

# Qwen3-0.6B: 28 layers × 8 kv_heads × 128 head_dim × 2 (K+V) × 2 bytes (fp16)
KV_BYTES_PER_TOKEN = 28 * 8 * 128 * 2 * 2


def make_cache(sink: int, window: int):
    """window<=0 -> baseline (full DynamicCache); else SinkWindowCache."""
    if window <= 0:
        return DynamicCache()
    return SinkWindowCache(sink_size=sink, window_size=window)


# ---------------------------------------------------------------------------
# 显存
# ---------------------------------------------------------------------------

def decode_peak_memory(model, device, context_ids, cache, max_new: int):
    """prefill(不计) → reset 峰值 → decode max_new 步 → 读峰值。

    基线与逐出用**相同**手写循环,仅 cache 不同 → 公平 A/B。返回
    (decode 峰值字节, cache 当前 KV 字节——解析精确,权威数字)。
    """
    with torch.no_grad():
        model(context_ids, past_key_values=cache, use_cache=True)  # prefill, 不测
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    input_ids = context_ids[:, -1:]
    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            input_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated(), kv_cache_bytes(cache)


# ---------------------------------------------------------------------------
# 质量: NIAH recall-by-position
# ---------------------------------------------------------------------------

def eval_niah(model, tok, device, context_with_needles, needles, sink, window, max_new: int):
    """每个 needle 一次 generate(past_key_values=SinkWindowCache)。

    prefill 全量上下文(能检索),decode 后中间被逐出 → 落在逐出区的 needle 掉 recall。
    问题永远在末尾窗口内,总能被看到。返回 per-position 结果 + 前后半段 recall。
    """
    results = []
    for n in needles:
        prompt = context_with_needles + "\n\n" + n["question"]
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tok(text, return_tensors="pt").to(device)
        cache = make_cache(sink, window)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, past_key_values=cache
            )
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        is_correct = n["answer"].lower() in resp.lower()
        results.append({"position": n["position"], "type": n["type"], "correct": is_correct})

    pos = compute_position_recall(results)
    return {
        "recall": pos["overall"],
        "front_half": pos["front_half"],
        "back_half": pos["back_half"],
        "results": results,
    }


# ---------------------------------------------------------------------------
# 质量(可选): perplexity
# ---------------------------------------------------------------------------

def measure_perplexity(model, tok, device, ids, sink, window, base_tokens: int, n_eval: int) -> float:
    """prefill 前 base_tokens 个,逐 token 累加 next-token NLL(shift-by-one)。

    窗口越小,长程一致性越差 → perplexity 越高。量化"丢了多少记忆"。
    """
    cache = make_cache(sink, window)
    base = ids[:, :base_tokens]
    with torch.no_grad():
        model(base, past_key_values=cache, use_cache=True)
    total_nll = 0.0
    count = 0
    end = min(base_tokens + n_eval, ids.shape[1] - 1)
    with torch.no_grad():
        for i in range(base_tokens, end):
            out = model(ids[:, i:i + 1], past_key_values=cache, use_cache=True)
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            total_nll -= logp[ids[0, i + 1]].item()  # logits 在 i 预测 i+1
            count += 1
    return math.exp(total_nll / count) if count else float("inf")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--quality-model", default=None,
                        help="Model used for the quality sections (NIAH/perplexity). "
                             "Defaults to --model. A stronger model (e.g. Qwen2.5-7B) is "
                             "needed for meaningful NIAH recall; the 0.6B ceiling is ~12-25%%.")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--windows", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--max-new", type=int, default=48,
                        help="Decode tokens per NIAH query (chat models are verbose; "
                             "16 was too short to surface the answer)")
    parser.add_argument("--num-needles", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--with-perplexity", action="store_true")
    parser.add_argument("--ppl-eval-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/kv_eviction_quality.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}  Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Quality model: a stronger retriever makes the eviction-vs-recall curve
    # visible (the 0.6B raw NIAH ceiling is ~12-25%, see run_realtext results).
    quality_name = args.quality_model or args.model
    if quality_name == args.model:
        quality_model, quality_tok = model, tok
    else:
        print(f"Quality model: {quality_name}")
        quality_model = AutoModelForCausalLM.from_pretrained(
            quality_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)
        quality_model.eval()
        quality_tok = AutoTokenizer.from_pretrained(quality_name, trust_remote_code=True)

    # ---- shared context ----
    book_text = load_real_text(args.book)
    context_text = prepare_real_context(book_text, args.context_tokens, tok)
    context_ids = tok(context_text, return_tensors="pt")["input_ids"].to(device)
    n_ctx = context_ids.shape[1]
    print(f"Context: {n_ctx} tokens  ({args.book})")

    context_with_needles, needles = insert_needles_mixed(
        context_text, args.num_needles, ["niah_single"], seed=args.seed
    )
    print(f"Needles: {len(needles)} (niah_single), positions: "
          f"{[round(n['position'], 2) for n in needles]}")

    # warmup (allocator / cudagraph noise)
    with torch.no_grad():
        model(context_ids[:, :64], use_cache=True)
    torch.cuda.empty_cache()

    measurements = {"memory": [], "quality_niah": [], "perplexity": []}
    configs = [("baseline", 0)] + [("window", w) for w in args.windows]

    # ---- 1. memory ----
    print("\n=== Memory (decode-phase) ===")
    for label, window in configs:
        sink = args.sink if window > 0 else 0
        peaks, kv_sizes = [], []
        for _ in range(args.runs):
            cache = make_cache(sink, window)
            peak, kv = decode_peak_memory(model, device, context_ids, cache, args.decode_tokens)
            peaks.append(peak)
            kv_sizes.append(kv)
            torch.cuda.empty_cache()
        peak = sorted(peaks)[len(peaks) // 2]
        kv = sorted(kv_sizes)[len(kv_sizes) // 2]
        keep = sink + window if window > 0 else n_ctx
        measurements["memory"].append({
            "label": label,
            "window": window,
            "keep_tokens": keep,
            "eviction_ratio": round(1 - keep / n_ctx, 4),
            "context_multiplier": round(n_ctx / keep, 1),
            "decode_peak_bytes": peak,
            "kv_bytes": kv,
            "kv_saved_bytes": None,  # 填在 summary 后处理
        })
        print(f"  {label:<9} window={window:<5} keep={keep} ratio={1 - keep / n_ctx:.3f} "
              f"mult={n_ctx / keep:.1f}x peak={peak / 1024**2:.0f}MB kv={kv / 1024**2:.0f}MB")

    # ---- 2. quality: NIAH recall-by-position ----
    print("\n=== Quality: NIAH recall-by-position ===")
    for label, window in configs:
        sink = args.sink if window > 0 else 0
        r = eval_niah(quality_model, quality_tok, device, context_with_needles, needles, sink, window, args.max_new)
        measurements["quality_niah"].append({
            "label": label, "window": window,
            "recall": r["recall"], "front_half": r["front_half"], "back_half": r["back_half"],
            "per_position": r["results"],
        })
        print(f"  {label:<9} window={window:<5} recall={r['recall']:.0%} "
              f"front={r['front_half']:.0%} back={r['back_half']:.0%}")
        torch.cuda.empty_cache()

    # ---- 3. optional perplexity ----
    if args.with_perplexity:
        print("\n=== Quality: perplexity (long-range memory) ===")
        ids = quality_tok(context_text, return_tensors="pt")["input_ids"].to(device)
        base = min(args.context_tokens, ids.shape[1] - args.ppl_eval_tokens - 1)
        for label, window in configs:
            sink = args.sink if window > 0 else 0
            ppl = measure_perplexity(quality_model, quality_tok, device, ids, sink, window, base, args.ppl_eval_tokens)
            measurements["perplexity"].append({"label": label, "window": window, "perplexity": round(ppl, 3)})
            print(f"  {label:<9} window={window:<5} ppl={ppl:.2f}")

    # ---- summary ----
    kv_base = measurements["memory"][0]["kv_bytes"]
    for m in measurements["memory"]:
        m["kv_saved_bytes"] = kv_base - m["kv_bytes"] if m["kv_bytes"] else None
    summary = {
        "model": args.model, "context_tokens": n_ctx,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "kv_bytes_baseline": kv_base,
        "per_window": [
            {
                "window": m["window"],
                "context_multiplier": m["context_multiplier"],
                "eviction_ratio": m["eviction_ratio"],
                "kv_saved_bytes": m["kv_saved_bytes"],
                "recall": next(q["recall"] for q in measurements["quality_niah"] if q["window"] == m["window"]),
                "perplexity": next(
                    (p["perplexity"] for p in measurements["perplexity"] if p["window"] == m["window"]),
                    None,
                ),
            }
            for m in measurements["memory"] if m["window"] > 0
        ],
        "note": "kv_bytes is the exact steady-state number and the authoritative memory metric; "
                "context_multiplier = context / (sink+window) is the 'same GPU, xN more context' "
                "headline. decode_peak_bytes barely moves because the prefill-phase KV is still "
                "resident when the decode peak counter is reset (eviction happens on the first "
                "decode step) — prefill phase peak is NOT saved, only decode steady-state KV is.",
    }

    results = {
        "config": {
            "model": args.model, "quality_model": quality_name, "book": args.book,
            "context_tokens": n_ctx, "sink": args.sink, "windows": args.windows,
            "max_new": args.max_new, "num_needles": args.num_needles, "runs": args.runs,
            "seed": args.seed, "with_perplexity": args.with_perplexity,
        },
        "measurements": measurements,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
