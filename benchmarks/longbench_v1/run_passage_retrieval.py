"""
LongBench v1 passage_retrieval_en 评估

任务：从多段落文档（30 段，~12K tokens）中，根据描述找出对应段落。
答案格式："Paragraph N"，accuracy 自动评分。

这是纯信息检索任务，0.6B 有能力做，框架压缩价值最大。

对比：truncation / project_topic / attention_sink / sink_topic
指标：段落号预测准确率

用法：.venv/bin/python benchmarks/longbench_v1/run_passage_retrieval.py
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "data")


def load_items(task: str = "passage_retrieval_en") -> List[Dict]:
    with open(os.path.join(DATA_DIR, f"{task}.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def choose_strategy(text: str, tokenizer, budget: int, strategy: str, query: str) -> str:
    if strategy == "truncation":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text
        return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)

    # 按段落切分（Paragraph N:）
    paras = re.split(r'(?=Paragraph \d+:)', text)
    paras = [p for p in paras if p.strip()]
    if not paras:
        return text[:budget] if len(text) > budget else text

    # 从 query 提取关键实体/词（支持中英文）
    query_words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', query)
                   if w.lower() not in {'the', 'that', 'this', 'these', 'those', 'with', 'from',
                                         'were', 'have', 'been', 'their', 'they', 'there', 'about',
                                         'text', 'summarizes', 'discusses', 'what', 'which', 'whose',
                                         'main', 'character', 'name', 'named', 'who', 'when', 'where'}]
    # 中文：提取 2-6 字词（排除常见虚词）
    cn_stopwords = {'一个', '什么', '如何', '关于', '根据', '描述', '下列', '其中', '哪些', '为什么',
                     '段落', '上面', '以下', '文本', '请', '回答', '找出', '匹配', '根据描述'}
    query_words += [w for w in re.findall(r'[\u4e00-\u9fa5]{2,6}', query)
                    if w not in cn_stopwords][:5]
    key = [p for p in paras if any(w.lower() in p.lower() for w in query_words)]
    other = [p for p in paras if p not in key]

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


def evaluate(model, tokenizer, device, items, strategy, budget, task="passage_retrieval_en", max_new=15) -> Dict:
    correct = 0
    times = []
    per_item = []

    for item in items:
        query = item["input"]
        ctx = choose_strategy(item["context"], tokenizer, budget, strategy, query)
        used_tok = len(tokenizer.encode(ctx, add_special_tokens=False))

        if task == "passage_count":
            # 数段落总数
            prompt = f"Passages:\n{ctx}\n\nHow many paragraphs are there in total? Answer with only the number:"
        elif task == "passage_retrieval_zh":
            prompt = f"段落列表：\n{ctx}\n\n找到与描述匹配的段落：{query}\n\n请回答段落编号（如 段落27）："
        else:
            prompt = f"Passages:\n{ctx}\n\nFind the passage that matches: {query}\n\nAnswer with the paragraph number (e.g., Paragraph 5):"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # 提取数字答案
        if task == "passage_retrieval_zh":
            m = re.search(r'段落\s*(\d+)', resp)
        else:
            m = re.search(r'(?:Paragraph|paragraph)?\s*(\d+)', resp)
        pred = m.group(1) if m else None
        gold = re.search(r'(\d+)', item["answers"][0]).group(1)
        is_correct = pred == gold
        if is_correct:
            correct += 1
        per_item.append({
            "id": item["_id"], "gold": gold, "pred": pred, "correct": is_correct,
            "full_tokens": len(tokenizer.encode(item["context"], add_special_tokens=False)),
            "used_tokens": used_tok,
        })

    return {
        "strategy": strategy, "budget": budget,
        "accuracy": correct / len(items), "correct": correct, "total": len(items),
        "avg_latency_s": round(sum(times) / len(times), 2),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--task", default="passage_retrieval_en",
                        choices=["passage_retrieval_en", "passage_retrieval_zh", "passage_count"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    items = load_items(task=args.task)[:args.samples]
    print(f"LongBench {args.task}: {len(items)} samples\n")
    if args.output is None:
        args.output = f"results/{args.task}_{args.model.split('/')[-1].replace('-Instruct','')}.json"

    results = {"config": {"model": args.model, "samples": len(items), "budgets": args.budgets}, "strategies": []}
    strategies = ["truncation", "project_topic", "attention_sink", "sink_topic"]

    for budget in args.budgets:
        print(f"=== Budget: {budget} tokens ===")
        for strat in strategies:
            r = evaluate(model, tokenizer, device, items, strat, budget, task=args.task)
            print(f"  {strat:<20} acc={r['accuracy']:.1%} ({r['correct']}/{r['total']})  latency={r['avg_latency_s']}s")
            results["strategies"].append(r)
        print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
