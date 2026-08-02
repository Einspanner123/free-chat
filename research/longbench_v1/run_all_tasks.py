"""
LongBench v1 多任务统一评估

跑多个 LongBench 任务，报告每个任务上框架策略 vs 截断的表现。
聚焦 0.6B 能力范围内的任务：检索/QA/分类（非深层推理、非生成摘要）。

任务分类：
- 信息检索类：passage_retrieval_en, passage_retrieval_zh, passage_count
- 单文档 QA：multifieldqa_en, multifieldqa_zh, qasper, narrativeqa
- 少样本/分类：triviaqa, trec, lsht
- 多文档 QA：hotpotqa, 2wikimqa（0.6B 可能到边界）

指标：accuracy（检索/分类）或 F1（QA）
用法：.venv/bin/python benchmarks/longbench_v1/run_all_tasks.py --task multifieldqa_en --samples 30
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "data")


def load_items(task: str) -> List[Dict]:
    with open(os.path.join(DATA_DIR, f"{task}.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def extract_query_words(query: str) -> List[str]:
    """提取 query 关键实体/词（支持中英文）。"""
    words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', query)
             if w.lower() not in {'what', 'which', 'where', 'when', 'how', 'why', 'who',
                                   'the', 'that', 'this', 'these', 'those', 'with', 'from',
                                   'were', 'have', 'been', 'their', 'they', 'there', 'about',
                                   'text', 'according', 'article', 'question', 'answer',
                                   'based', 'following', 'passage', 'main', 'character',
                                   'summarizes', 'discusses', 'whose', 'name', 'named'}]
    cn_stop = {'一个', '什么', '如何', '关于', '根据', '描述', '下列', '其中', '哪些',
               '为什么', '上面', '以下', '文本', '请', '回答', '找出', '匹配', '请根据'}
    words += [w for w in re.findall(r'[\u4e00-\u9fa5]{2,6}', query) if w not in cn_stop][:5]
    return words


def choose_strategy(text: str, tokenizer, budget: int, strategy: str, query: str) -> str:
    if strategy == "truncation":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text
        return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)

    # 按句/段落分块
    sentences = re.split(r'(?<=[.!?。！？])\s*', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return text[:budget] if len(text) > budget else text

    query_words = extract_query_words(query)
    # 优化 A: 按 query 词命中数给段落打分，取最相关的 top_k 段
    scored = []
    for s in sentences:
        hits = sum(1 for w in query_words if w.lower() in s.lower())
        if hits > 0:
            scored.append((hits, s))
    # 按命中数降序，同分保持原文顺序
    scored.sort(key=lambda x: -x[0])
    top_k = 3
    key = [s for _, s in scored[:top_k]]
    other = [s for s in sentences if s not in key]

    if strategy == "project_topic":
        result = list(key)
        total = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in result)
        tail = []
        for i, s in enumerate(reversed(other)):
            turn = i + 1
            ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
            if not ct:
                continue
            nt = len(tokenizer.encode(ct, add_special_tokens=False))
            if total + nt <= budget:
                tail.insert(0, ct)
                total += nt
        return " ".join(result + tail)

    elif strategy == "attention_sink":
        key_text = " ".join(key)
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

    elif strategy == "sink_topic":
        key_text = "\n\n".join(key)
        key_tok = len(tokenizer.encode(key_text, add_special_tokens=False))
        remaining = budget - key_tok - 2
        compressed = []
        if remaining > 0:
            for i, s in enumerate(reversed(other)):
                turn = i + 1
                ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
                if not ct:
                    continue
                nt = len(tokenizer.encode(ct, add_special_tokens=False))
                if sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in compressed) + nt <= remaining:
                    compressed.insert(0, ct)
        return "\n\n" + key_text + "\n\n" + " ".join(compressed)

    return text


def compute_f1(pred: str, ref: str) -> float:
    p = set(pred.lower().split())
    r = set(ref.lower().split())
    if not p or not r:
        return 0.0
    common = p & r
    prec = len(common) / len(p)
    rec = len(common) / len(r)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def evaluate(model, tokenizer, device, items, strategy, budget, task, max_new=30) -> Dict:
    correct = 0
    total_f1 = 0
    times = []
    per_item = []
    is_accuracy_task = task in ("passage_retrieval_en", "passage_retrieval_zh", "passage_count", "trec", "lsht")

    for item in items:
        query = item["input"]
        ctx = choose_strategy(item["context"], tokenizer, budget, strategy, query)
        used_tok = len(tokenizer.encode(ctx, add_special_tokens=False))

        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gold = item["answers"][0] if isinstance(item["answers"], list) else item["answers"]

        if is_accuracy_task:
            # 提取数字/类别
            if task in ("passage_retrieval_en", "passage_count"):
                m = re.search(r'(?:Paragraph|paragraph)?\s*(\d+)', resp)
                pred = m.group(1) if m else None
            elif task == "passage_retrieval_zh":
                m = re.search(r'段落\s*(\d+)', resp)
                pred = m.group(1) if m else None
            else:  # trec, lsht
                pred = resp.strip()
            g = re.search(r'(\d+)', str(gold)).group(1) if re.search(r'(\d+)', str(gold)) else str(gold).strip()
            is_correct = (pred == g) if pred else False
            if is_correct:
                correct += 1
        else:
            # QA 用 F1
            f1 = compute_f1(resp, str(gold))
            total_f1 += f1
            is_correct = f1 > 0.5
            if is_correct:
                correct += 1

        per_item.append({
            "id": item.get("_id", ""), "gold": str(gold)[:40], "pred": resp.strip()[:40],
            "correct": is_correct,
            "full_tokens": len(tokenizer.encode(item["context"], add_special_tokens=False)),
            "used_tokens": used_tok,
        })

    n = len(items)
    metric = correct / n if is_accuracy_task else (total_f1 / n if n else 0)
    return {
        "strategy": strategy, "budget": budget, "task": task,
        "metric": round(metric, 4), "correct": correct, "total": n,
        "metric_type": "accuracy" if is_accuracy_task else "f1",
        "avg_latency_s": round(sum(times) / len(times), 2),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--task", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Task: {args.task}, Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    items = load_items(args.task)[:args.samples]
    print(f"Samples: {len(items)}\n")

    if args.output is None:
        model_short = args.model.split("/")[-1].replace("-Instruct", "")
        args.output = f"results/{args.task}_{model_short}.json"

    results = {"config": {"model": args.model, "task": args.task, "samples": len(items), "budgets": args.budgets}, "strategies": []}
    strategies = ["truncation", "project_topic", "attention_sink", "sink_topic"]

    for budget in args.budgets:
        print(f"=== Budget: {budget} ===")
        for strat in strategies:
            r = evaluate(model, tokenizer, device, items, strat, budget, args.task)
            mtype = r["metric_type"]
            print(f"  {strat:<20} {mtype}={r['metric']:.3f} ({r['correct']}/{r['total']})")
            results["strategies"].append(r)
        print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
