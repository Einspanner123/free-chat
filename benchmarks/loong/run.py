"""
Loong 中文长上下文多文档 QA Benchmark 评估

Loong (EMNLP 2024): 多文档 QA，平均 11 个文档，上下文 10K-200K+ tokens
任务类型：Spotlight Locating / Comparison / Clustering / Chain of Reasoning

评估：Qwen3-0.6B + 上下文压缩策略 vs 直接截断
对比指标：答案中关键实体的召回（简化版，因为 0.6B 无法完整完成 GPT-4 评估）

用法：.venv/bin/python benchmarks/loong/run.py --samples 8
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Loong 官方 leaderboard（Overall）
PUBLISHED_BASELINES = {
    "Gemini-1.5-pro": 55.37,
    "GPT-4o": 53.47,
    "Claude-3.5-Sonnet": 48.85,
    "Qwen2-72B": 40.71,
    "Llama-3.1-8B": 25.43,
    "GPT-3.5": 22.0,
}


def load_docs() -> Dict[str, str]:
    """加载所有文档。"""
    docs = {}
    for domain in ["legal", "financial", "paper"]:
        json_path = os.path.join(DATA_DIR, "doc", domain, f"{domain}.json")
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                docs.update(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "title" in item:
                        docs[item["title"]] = item
    return docs


def load_items() -> List[Dict]:
    with open(os.path.join(DATA_DIR, "loong.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def assemble_context(item: Dict, docs: Dict[str, str]) -> str:
    """按 prompt_template 组装上下文。"""
    doc_texts = []
    for dname in item["doc"]:
        d = docs.get(dname)
        if d is None:
            continue
        if isinstance(d, dict):
            content = d.get("content", str(d))
        else:
            content = str(d)
        # 中文文书：doc 字段是标题，内容在 legal.json 的 key 里
        doc_texts.append(f"<di> {content}")
    return "\n".join(doc_texts)


def choose_strategy(text: str, tokenizer, budget: int, strategy: str, question: str) -> str:
    """压缩上下文到预算。"""
    if strategy == "truncation":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text
        return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)

    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    if not sentences:
        return text[:budget] if len(text) > budget else text

    # 从问题提取关键实体
    question_words = [w for w in re.findall(r'[\u4e00-\u9fa5]{2,}', question) if len(w) > 1]
    key = [s for s in sentences if any(w in s for w in question_words[:5])]
    other = [s for s in sentences if s not in key]

    if strategy == "project_topic":
        result = list(key)
        total = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in result)
        for i, s in enumerate(reversed(other)):
            turn = i + 1
            ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
            if not ct:
                continue
            nt = len(tokenizer.encode(ct, add_special_tokens=False))
            if total + nt <= budget:
                result.insert(0, ct)
                total += nt
        return " ".join(result)

    elif strategy == "attention_sink":
        key_text = "\n\n".join(key)
        key_tok = len(tokenizer.encode(key_text, add_special_tokens=False))
        remaining = budget - key_tok - 2
        compressed = []
        if remaining > 0:
            for i, s in enumerate(reversed(other)):
                ct = s if i < 5 else (s[:100] if i < 20 else (s[:50] if i < 50 else ""))
                if not ct:
                    continue
                nt = len(tokenizer.encode(ct, add_special_tokens=False))
                if sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in compressed) + nt <= remaining:
                    compressed.insert(0, ct)
        return "\n\n" + key_text + "\n\n" + " ".join(compressed)

    return text


def evaluate(model, tokenizer, device, items, docs, strategy, budget, max_new=100) -> Dict:
    correct = 0
    times = []
    per_item = []
    total_tokens = 0

    for item in items:
        full_ctx = assemble_context(item, docs)
        if not full_ctx:
            continue
        ctx = choose_strategy(full_ctx, tokenizer, budget, strategy, item["question"])

        instruction = item["instruction"]
        prompt = f"#Papers:\n{ctx}\n\n{instruction}\n\n#The paper you need to analyze:\n{item['question']}"

        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        times.append(dt)
        total_tokens += len(tokenizer.encode(ctx, add_special_tokens=False))

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        per_item.append({
            "id": item["id"], "question": item["question"][:50],
            "response_len": len(resp), "response_head": resp[:100],
            "full_ctx_tokens": len(tokenizer.encode(full_ctx, add_special_tokens=False)),
            "used_tokens": len(tokenizer.encode(ctx, add_special_tokens=False)),
        })

    return {
        "strategy": strategy, "budget": budget,
        "avg_latency_s": round(sum(times) / len(times), 2),
        "avg_response_len": round(sum(p["response_len"] for p in per_item) / max(len(per_item), 1)),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--output", default="results/loong_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    docs = load_docs()
    items = load_items()
    # 选中文样本
    zh = [i for i in items if i["language"] == "zh"]
    selected = zh[:args.samples]
    print(f"Loong benchmark: {len(zh)} 中文样本, selected {len(selected)}")
    print(f"Documents loaded: {len(docs)}\n")

    results = {
        "config": {"model": args.model, "samples": len(selected), "budgets": args.budgets},
        "published_baselines": PUBLISHED_BASELINES,
        "strategies": [],
    }

    strategies = ["truncation", "project_topic", "attention_sink"]

    for budget in args.budgets:
        print(f"=== Budget: {budget} tokens ===")
        for strat in strategies:
            r = evaluate(model, tokenizer, device, selected, docs, strat, budget)
            print(f"  {strat:<20} latency={r['avg_latency_s']}s  avg_resp={r['avg_response_len']} chars")
            results["strategies"].append(r)
        print()

    # 展示每个策略的回答样例对比
    print("=== Response comparison (budget 4096) ===")
    strat_responses = {}
    for r in results["strategies"]:
        if r["budget"] == args.budgets[0]:
            strat_responses[r["strategy"]] = r["per_item"]

    if len(args.budgets) > 0:
        sample_idx = 0
        if sample_idx < len(selected):
            q = selected[sample_idx]["question"][:80]
            print(f"\nQuestion: {q}\n")
            for strat in strategies:
                item = strat_responses.get(strat, [])[sample_idx]
                print(f"--- {strat} (used {item['used_tokens']} tok) ---")
                print(item["response_head"][:150])
                print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
