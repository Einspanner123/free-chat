"""
Attention temperature (t) ablation on LongBench passage_retrieval_en.

Motivation: YaRN scales attention logits by a temperature factor t
before softmax: softmax(QK^T / (sqrt(d) * t)). This sharpens attention
on long contexts. We inject t by patching eager_attention_forward.

Setup:
- Context: BM25 top-1 paragraph (framework best strategy)
- Model: Qwen3-0.6B, forced eager attention (to access attn_weights)
- Variable: attention temperature t {0.5, 1.0, 2.0, 4.0}
  (t < 1 sharpens, t > 1 softens attention)
- Metric: paragraph retrieval accuracy, greedy decoding (no sampling noise)

Usage: .venv/bin/python research/longbench_v1/run_attn_temp_ablation.py
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse context-engine BM25 pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "services", "context-engine", "src"))
from pipeline import ContextPipeline, PipelineConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "data")

# Global attention temperature (patched into eager_attention_forward)
ATTN_TEMPERATURE = 1.0


def patched_eager_attention_forward(
    module,
    query, key, value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    """Eager attention with attention temperature scaling (YaRN-style)."""
    global ATTN_TEMPERATURE
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    # Apply attention temperature: attn_weights = QK^T * scaling / t
    effective_scaling = scaling / ATTN_TEMPERATURE
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * effective_scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep > 1:
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    return hidden_states


def load_items() -> List[Dict]:
    with open(os.path.join(DATA_DIR, "passage_retrieval_en.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def evaluate(model, tokenizer, device, items, attn_temp, budget=2048) -> Dict:
    global ATTN_TEMPERATURE
    ATTN_TEMPERATURE = attn_temp

    pipe = ContextPipeline(PipelineConfig(strategy="bm25_top1", budget=budget, retriever="bm25", top_k=1))
    correct = 0
    times = []
    per_item = []

    for item in items:
        ctx = pipe.build(item["context"], tokenizer, query=item["input"])
        prompt = f"Passages:\n{ctx}\n\nFind the passage that matches: {item['input']}\n\nAnswer with the paragraph number:"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)  # greedy, no sampling noise
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = re.search(r'(\d+)', resp)
        pred_n = pred.group(1) if pred else None
        gold = re.search(r'(\d+)', item["answers"][0]).group(1)
        is_correct = pred_n == gold
        if is_correct:
            correct += 1
        per_item.append({"gold": gold, "pred": pred_n, "correct": is_correct})

    return {
        "attn_temperature": attn_temp,
        "accuracy": correct / len(items),
        "correct": correct, "total": len(items),
        "avg_latency_s": round(sum(times) / len(times), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--attn-temps", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--output", default="results/attn_temp_ablation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")

    # Load with FORCED eager attention (to access attn_weights)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Monkey-patch eager_attention_forward with temperature scaling
    import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
    qwen3_modeling.eager_attention_forward = patched_eager_attention_forward
    print("Patched eager_attention_forward with attention temperature\n")

    items = load_items()[:args.samples]
    print(f"passage_retrieval_en: {len(items)} samples\n")

    results = {"config": {"model": args.model, "samples": len(items), "attn_temperatures": args.attn_temps}, "runs": []}

    # Baseline first (t=1.0, no modification)
    for temp in args.attn_temps:
        r = evaluate(model, tokenizer, device, items, temp)
        label = "baseline" if temp == 1.0 else f"t={temp}"
        print(f"  attn_t={temp:>4}: acc={r['accuracy']:.1%} ({r['correct']}/{r['total']})  {label}")
        results["runs"].append(r)

    ATTN_TEMPERATURE = 1.0  # reset
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
