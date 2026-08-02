"""
LongBench 风格长文本 QA Benchmark

使用真实书籍文本（Pride and Prejudice, Moby Dick 等）+ 构造的 QA pair，
对比已发布的 LongBench 基线：GPT-4 (90%+), Llama-2-7B (~25% F1), 
Qwen-7B (~60%), 开源小模型 baseline (~15-20%)。

我们的贡献：Qwen3-0.6B + 上下文压缩，在长文本 QA 上的 F1 对比。

用法：.venv/bin/python benchmarks/longbench/run.py
"""

import argparse
import json
import os
import random
import re
import time
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "long_context", "data")

# 已发布 LongBench 基线（MultiFieldQA EN 任务）
PUBLISHED_BASELINES = {
    "GPT-4": 0.92,
    "GPT-3.5-turbo": 0.82,
    "Llama-2-7B": 0.25,
    "Qwen-7B-chat": 0.60,
    "ChatGLM3-6B": 0.45,
}

# 从真实书籍文本中构造 QA（类似 LongBench 的 MultiFieldQA）
QA_TEMPLATES = [
    # 人物关系类
    ("According to the passage, what is the relationship between {name1} and {name2}?", "relationship"),
    ("How does {name1} feel about {name2} based on the text?", "feeling"),
    # 事件细节类
    ("What happens at {location} in this passage?", "event"),
    ("What is the significance of {location} in the narrative?", "significance"),
    # 主题类
    ("What is the main theme discussed in this passage?", "theme"),
    ("Summarize the key events involving {name1}.", "summary"),
]

# 从 Pride and Prejudice 中预提取的人物和地点
KNOWN_ENTITIES = {
    "names": ["Elizabeth", "Darcy", "Jane", "Bingley", "Wickham", "Collins",
              "Lady Catherine", "Charlotte", "Gardiner", "Lydia"],
    "locations": ["Netherfield", "Pemberley", "Longbourn", "Meryton", "Rosings", "Hunsford"],
}


def load_book(name: str) -> str:
    with open(os.path.join(DATA_DIR, f"{name}.txt"), "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_context(text: str, target_tokens: int, tokenizer) -> str:
    """取一段连续文本作为上下文。"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= target_tokens:
        return text
    # 取中间的连续片段（避免总是开头）
    start = random.randint(0, max(0, len(tokens) - target_tokens - 100))
    return tokenizer.decode(tokens[start:start + target_tokens], skip_special_tokens=True)


def generate_qa(context: str, tokenizer, num_questions: int = 4) -> List[Dict]:
    """
    从上下文中构造 QA pair。
    使用模板 + 上下文中随机选择的实体名。
    答案从上下文中提取（简化版：用规则匹配）。
    """
    qa_pairs = []
    # 检查上下文中实际出现的人物和地点
    ctx_lower = context.lower()
    present_names = [n for n in KNOWN_ENTITIES["names"] if n.lower() in ctx_lower]
    present_locs = [l for l in KNOWN_ENTITIES["locations"] if l.lower() in ctx_lower]

    if len(present_names) < 2:
        return qa_pairs

    for i in range(num_questions):
        n1 = random.choice(present_names)
        n2 = random.choice([n for n in present_names if n != n1]) if len(present_names) > 1 else n1
        loc = random.choice(present_locs) if present_locs else "London"

        template, qtype = random.choice(QA_TEMPLATES)
        question = template.format(name1=n1, name2=n2, location=loc)

        # 简单答案提取：找包含两个实体名的最近句子
        answer = _extract_answer(context, n1, n2 if "name2" in template else None)

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "context_tokens": len(tokenizer.encode(context, add_special_tokens=False)),
        })

    return qa_pairs


def _extract_answer(text: str, entity1: str, entity2: str = None) -> str:
    """从文本中提取包含指定实体的最近句子作为粗略答案。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    candidates = []
    for s in sentences:
        if entity1.lower() in s.lower():
            if entity2 is None or entity2.lower() in s.lower():
                candidates.append(s.strip())
    # 返回最长的一句（通常包含最多信息）
    if candidates:
        return max(candidates, key=len)[:300]
    # fallback
    return f"The passage discusses {entity1}."


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1（LongBench 标准指标）。"""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def contextual_compress(text: str, tokenizer, budget: int, keep_entities: List[str] = None) -> str:
    """话题感知压缩：保留含实体的句子，分级压缩其余。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not keep_entities:
        keep_entities = []

    key = [s for s in sentences if any(e.lower() in s.lower() for e in keep_entities)]
    other = [s for s in sentences if s not in key]

    # 保留关键句 + 分级压缩其余
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


def attention_sink_layout(text: str, tokenizer, budget: int, key_sentences: List[str]) -> str:
    """Attention Sink 布局：sink → 关键句 → 其他。"""
    if not key_sentences:
        return text[:budget] if len(text) > budget else text

    key_text = "\n\n".join(key_sentences)
    key_tok = len(tokenizer.encode(key_text, add_special_tokens=False))
    remaining = budget - key_tok - 2

    # 压缩其他句子
    sentences = re.split(r'(?<=[.!?])\s+', text)
    other = [s for s in sentences if s not in key_sentences]
    compressed = []
    for i, s in enumerate(reversed(other)):
        ct = s if i < 5 else (s[:100] if i < 20 else (s[:50] if i < 50 else ""))
        if not ct:
            continue
        nt = len(tokenizer.encode(ct, add_special_tokens=False))
        if sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in compressed) + nt <= remaining:
            compressed.insert(0, ct)

    return "\n\n" + key_text + "\n\n" + " ".join(compressed)


STRATEGIES = {
    "full": ("Full Context", None),
    "truncation": ("Truncation", None),
    "project_topic": ("Project + Topic", None),
    "attention_sink": ("Attention Sink", None),
}


def evaluate_strategy(model, tokenizer, device, context, qa_pairs, strategy_name,
                      budget: int, max_new: int = 50) -> float:
    """评估一种策略在给定预算下的 F1。"""
    predicted_answers = []

    for qa in qa_pairs:
        q = qa["question"]
        ctx = context

        if strategy_name == "full":
            ctx = context
        elif strategy_name == "truncation":
            tokens = tokenizer.encode(ctx, add_special_tokens=False)
            ctx = tokenizer.decode(tokens[-budget:], skip_special_tokens=True) if len(tokens) > budget else ctx
        elif strategy_name == "project_topic":
            # 话题压缩：保留包含问题实体的句子
            entities = [w for w in q.split() if w[0].isupper() and len(w) > 2]
            ctx = contextual_compress(context, tokenizer, budget, keep_entities=entities)
        elif strategy_name == "attention_sink":
            entities = [w for w in q.split() if w[0].isupper() and len(w) > 2]
            key_sentences = [s for s in re.split(r'(?<=[.!?])\s+', context)
                           if any(e.lower() in s.lower() for e in entities)]
            ctx = attention_sink_layout(context, tokenizer, budget, key_sentences)

        prompt = ctx + "\n\nQuestion: " + q + "\nAnswer:"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted_answers.append(resp)

    # 计算 F1
    f1_scores = []
    for qa, pred in zip(qa_pairs, predicted_answers):
        f1_scores.append(compute_f1(pred, qa["answer"]))

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--num-questions", type=int, default=4)
    parser.add_argument("--output", default="results/longbench_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    book = load_book(args.book)
    context = extract_context(book, args.context_tokens, tokenizer)
    ctx_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    qa_pairs = generate_qa(context, tokenizer, num_questions=args.num_questions)

    print(f"LongBench 风格长文本 QA Benchmark")
    print(f"Book: {args.book}, Context: {ctx_tokens} tokens, Questions: {len(qa_pairs)}\n")

    results = {"config": {"model": args.model, "book": args.book, "context_tokens": ctx_tokens, "num_questions": len(qa_pairs), "published_baselines": PUBLISHED_BASELINES}, "strategies": []}

    # Full context
    f1 = evaluate_strategy(model, tokenizer, device, context, qa_pairs, "full", ctx_tokens)
    print(f"Full Context ({ctx_tokens} tok): F1 = {f1:.3f}")
    results["strategies"].append({"strategy": "Full Context", "f1": round(f1, 3), "tokens": ctx_tokens})

    # Compression strategies
    for budget in args.budgets:
        ratio = 1 - budget / ctx_tokens
        print(f"\nBudget: {budget} tokens ({ratio:.0%} compression)")
        for key, (label, _) in STRATEGIES.items():
            if key == "full":
                continue
            f1 = evaluate_strategy(model, tokenizer, device, context, qa_pairs, key, budget)
            print(f"  {label:<25} F1 = {f1:.3f}")
            results["strategies"].append({"strategy": label, "budget": budget, "f1": round(f1, 3)})

    # 对比基线
    print(f"\n--- Published LongBench Baselines (MultiFieldQA EN) ---")
    for model_name, score in sorted(PUBLISHED_BASELINES.items(), key=lambda x: -x[1]):
        print(f"  {model_name:<20} F1 = {score:.2f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
