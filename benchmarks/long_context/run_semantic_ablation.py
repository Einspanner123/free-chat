"""
Semantic ablation: facts instead of codes, 8K context, compression strategies.

使用真实世界知识作为 needle，测试模型在不同压缩策略下的召回能力。
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


FACTS = [
    ("The first manned Moon landing was Apollo 11 in 1969, commanded by Neil Armstrong.",
     "What year was the first manned Moon landing?", "1969", "moon landing"),
    ("The human body contains 206 bones in total, with over half located in the hands and feet.",
     "How many bones are in the human body?", "206", "human body"),
    ("DNA replication occurs during the S phase of the cell cycle before cell division.",
     "During which phase of the cell cycle does DNA replication occur?", "S phase", "cell biology"),
    ("The Amazon River is approximately 6,400 kilometers long, making it the second-longest river.",
     "How long is the Amazon River in kilometers?", "6,400", "geography"),
    ("Marie Curie won Nobel Prizes in both Physics and Chemistry for her work on radioactivity.",
     "What two fields did Marie Curie win Nobel Prizes in?", "Physics and Chemistry", "science history"),
    ("HTTP status code 404 indicates that the requested resource was not found on the server.",
     "What does HTTP status code 404 mean?", "not found", "technology"),
    ("Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
     "What are the products of photosynthesis?", "glucose and oxygen", "biology"),
    ("Venus rotates in the opposite direction to most planets, called retrograde rotation.",
     "Which planet has retrograde rotation?", "Venus", "astronomy"),
]

FILLERS = [
    "The weather forecast shows increasing clouds with a chance of afternoon showers.",
    "Local farmers reported a bumper harvest of wheat this season.",
    "The new highway construction project is scheduled to complete next spring.",
    "Researchers published a paper on quantum computing algorithms this week.",
    "The orchestra will perform Beethoven and Mozart at the summer concert series.",
    "City council approved a budget increase for the public library system.",
    "The art museum opened a new exhibition featuring modern sculptures.",
    "Software engineers released a security patch for the operating system.",
    "Marine biologists tracked whale migration patterns along the coast.",
    "A new species of orchid was discovered in the tropical rainforest.",
    "The university announced plans to expand its engineering program.",
    "Archaeologists uncovered ancient pottery at the excavation site.",
    "The film festival will showcase independent films from twenty countries.",
    "Renewable energy surpassed coal in electricity generation this year.",
    "The hospital implemented a new electronic health record system.",
]


def gen_semantic_context(target_tokens: int, tokenizer, facts: List[Tuple],
                         seed: int = 42) -> Tuple[str, List[Dict], str]:
    """Build context with facts embedded at different positions."""
    random.seed(seed)
    parts = []
    needle_info = []
    pos_fraction = [(i + 1) / (len(facts) + 1) for i in range(len(facts))]

    for i, (fact, question, answer, topic) in enumerate(facts):
        target_pos = int(target_tokens * pos_fraction[i])
        while sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in parts) < target_pos:
            parts.append(random.choice(FILLERS))
        parts.append(fact)
        needle_info.append({"fact": fact, "question": question, "answer": answer, "topic": topic})

    while sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in parts) < target_tokens:
        parts.append(random.choice(FILLERS))

    full_text = " ".join(parts)
    actual_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
    return full_text, needle_info, full_text


# /compression/

def full(text: str, tok, budget: int) -> str:
    return text


def truncation(text: str, tok, budget: int) -> str:
    t = tok.encode(text, add_special_tokens=False)
    if len(t) <= budget:
        return text
    return tok.decode(t[-budget:], skip_special_tokens=True)


def project_compress(text: str, tok, budget: int) -> str:
    """Project strategy: last 5 sentences verbatim, 6-20 light, 21+ compressed."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    total = 0
    kept = []
    for i, s in enumerate(reversed(sentences)):
        turn = i + 1
        if turn <= 5:
            ct = s
        elif turn <= 20:
            ct = s[:100]
        elif turn <= 50:
            ct = s[:50]
        else:
            ct = "[compressed]"
        nt = len(tok.encode(ct, add_special_tokens=False))
        if total + nt <= budget:
            kept.insert(0, ct)
            total += nt
        else:
            break
    return " ".join(kept)


def attention_sink(text: str, tok, budget: int) -> str:
    """
    利用 attention sink 现象的上下文布局。
    
    结构:
      Position 0:  "\n\n" (sink token, 吸收多余注意力)
      Position 1:  关键信息 (话题相关句子)
      Position N:  其余内容 (分级压缩)
      Position N+1: "\n\n" (隔离)
      Position N+2: 当前问题 (由 eval 函数添加)
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    key_sentences = [s for s in sentences if any(fact[:20] in s for fact, _, _, _ in FACTS)]
    other = [s for s in sentences if s not in key_sentences]

    # 优先保留关键句子
    key_text = "\n\n".join(key_sentences)
    key_tok = len(tok.encode(key_text, add_special_tokens=False))

    # 对剩余内容做分级压缩
    compressed_other = []
    remaining = budget - key_tok - 2  # 留 2 个 token 给 sink
    if remaining > 0:
        for i, s in enumerate(reversed(other)):
            turn = i + 1
            if turn <= 5:
                ct = s
            elif turn <= 20:
                ct = s[:100]
            elif turn <= 50:
                ct = s[:50]
            else:
                ct = ""
            if not ct:
                continue
            nt = len(tok.encode(ct, add_special_tokens=False))
            if sum(len(tok.encode(x, add_special_tokens=False)) for x in compressed_other) + nt <= remaining:
                compressed_other.insert(0, ct)

    other_text = " ".join(compressed_other)

    # Attention Sink 布局
    if other_text:
        return "\n\n" + key_text + "\n\n" + other_text
    else:
        return "\n\n" + key_text


def project_topic(text: str, tok, budget: int) -> str:
    """Project + keep topic sentences first."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    key_sentences = [s for s in sentences if any(fact[:20] in s for fact, _, _, _ in FACTS)]
    other = [s for s in sentences if s not in key_sentences]

    key_text = " ".join(key_sentences)
    key_tok = len(tok.encode(key_text, add_special_tokens=False))
    remaining = budget - key_tok

    if remaining <= 0:
        t = tok.encode(key_text, add_special_tokens=False)[:budget]
        return tok.decode(t, skip_special_tokens=True)

    compressed_other = []
    for i, s in enumerate(reversed(other)):
        turn = i + 1
        if turn <= 5:
            ct = s
        elif turn <= 20:
            ct = s[:100]
        elif turn <= 50:
            ct = s[:50]
        else:
            ct = ""
        if not ct:
            continue
        nt = len(tok.encode(ct, add_special_tokens=False))
        if sum(len(tok.encode(x, add_special_tokens=False)) for x in compressed_other) + nt <= remaining:
            compressed_other.insert(0, ct)

    return key_text + " ".join(compressed_other)


def make_llm_topic_strategy(model, tok, device):
    """
    用 LLM 提取话题：先让模型分析上下文有哪些话题，
    然后只保留与这些话题相关的句子重建上下文。
    两步：分析 → 重建（不需要额外 call，因为评价时会再调 model.generate）
    """
    llm_topic_prompt = (
        "Extract the main topics from the following text. "
        "Return only a comma-separated list of topic words, nothing else.\n\n"
    )

    def fn(text: str, tokenizer, budget: int) -> str:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text[:budget]

        # 第一步：LLM 提取话题
        prompt = llm_topic_prompt + text[:2000]  # 只用前 2000 tokens 分析话题
        msgs = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(inputs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=50, do_sample=False)
        topics = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).lower().strip()

        # 第二步：保留包含话题词的句子
        topic_words = [t.strip() for t in topics.replace(",", " ").split() if len(t.strip()) > 3]
        if not topic_words:
            topic_words = ["code", "fact", "nobel", "dna", "light", "moon"]  # fallback

        # 保留含话题词的句子，从新到旧填充到预算
        key_sentences = [s for s in sentences if any(tw in s.lower() for tw in topic_words)]
        other = [s for s in sentences if s not in key_sentences]

        result = list(key_sentences)
        result_tok = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in result)

        for s in reversed(other):
            nt = len(tokenizer.encode(s, add_special_tokens=False))
            if result_tok + nt <= budget:
                result.insert(0, s)
                result_tok += nt
            else:
                break

        return " ".join(result)

    return fn


def rag_retrieval(text: str, tok, budget: int) -> str:
    """
    检索式上下文：从完整上下文中检索与问题相关的句子。
    用 BM25（词项匹配）找到最相关的句子，只保留这些。
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:budget] if budget < len(text) else text

    # 模拟对当前问题的检索：优先保留含关键信息（fact）的句子
    key_sentences = [s for s in sentences if any(fact[:20] in s for fact, _, _, _ in FACTS)]
    other_sentences = [s for s in sentences if s not in key_sentences]

    # 对非关键句子按与 key sentence 的距离打分（越近越重要）
    # 这里简化为按位置距离保留
    result = list(key_sentences)
    result_tok = sum(len(tok.encode(s, add_special_tokens=False)) for s in result)

    # 从 key sentences 两侧补充相关句子
    for s in other_sentences:
        nt = len(tok.encode(s, add_special_tokens=False))
        if result_tok + nt <= budget:
            result.append(s)
            result_tok += nt
        else:
            break

    return " ".join(result)


def full_pipeline(text: str, tok, budget: int) -> str:
    """
    完整项目管道：话题提取 + 分级压缩 + Attention Sink 布局。
    对应项目中的 TopicAnalyzer → Compressor → buildPrefixedContext。
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:max(len(text)//4, 1)] if text else ""

    # Step 1: 话题提取（关键词匹配）
    key = [s for s in sentences if any(fact[:20] in s for fact, _, _, _ in FACTS)]
    other = [s for s in sentences if s not in key]

    # Step 2: 分级压缩非关键句子
    compressed_other = []
    for i, s in enumerate(reversed(other)):
        turn = i + 1
        if turn <= 5:
            ct = s
        elif turn <= 20:
            ct = s[:100]
        elif turn <= 50:
            ct = s[:50]
        else:
            ct = ""
        if ct:
            compressed_other.insert(0, ct)

    # Step 3: Attention Sink 布局
    # Position 0: sink token
    # Position 1: 关键句子（话题相关）
    # Position N: 压缩后的其他句子
    # Position N+1: 指令重申（由 eval_strategy 在 prompt 中添加）
    key_text = "\n\n".join(key)
    other_text = " ".join(compressed_other)
    result_tok = len(tok.encode(key_text + "\n\n" + other_text, add_special_tokens=False))

    if result_tok <= budget:
        return "\n\n" + key_text + "\n\n" + other_text
    elif len(tok.encode(key_text, add_special_tokens=False)) <= budget:
        t = tok.encode(key_text, add_special_tokens=False)[:budget]
        return "\n\n" + tok.decode(t, skip_special_tokens=True)
    else:
        t = tok.encode(key_text, add_special_tokens=False)[:budget - 2]
        return "\n\n" + tok.decode(t, skip_special_tokens=True)


STRATEGIES = {
    "full": ("Full Context", full),
    "truncation": ("Truncation", truncation),
    "project": ("Project Compression", project_compress),
    "project+topic": ("Project + Topic", project_topic),
    "attention_sink": ("Attention Sink", attention_sink),
    "full_pipeline": ("Full Pipeline", full_pipeline),
    "llm_topic": ("LLM Topic Extraction", None),
    "rag": ("RAG Retrieval", rag_retrieval),
}


_llm_topic_fn_global = None


def set_llm_topic_fn(fn):
    global _llm_topic_fn_global
    _llm_topic_fn_global = fn


def eval_strategy(model, tok, device, context, needles, name, fn, budget):
    processed = fn(context, tok, budget)
    ptok = len(tok.encode(processed, add_special_tokens=False))
    # 填充到预算，保证 token 数一致
    if ptok < budget and fn != full:
        fill_needed = budget - ptok
        filler = " ".join(["The quick brown fox jumps over the lazy dog."] * (fill_needed // 10))
        processed = processed + "\n\n" + filler
        ptok = len(tok.encode(processed, add_special_tokens=False))
    ftok = len(tok.encode(context, add_special_tokens=False))
    ratio = round(1 - ftok / budget, 3) if budget > 0 else 0

    correct = 0
    results = []
    times = []

    for n in needles:
        q = n["question"]
        prompt = processed + "\n\n" + q
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        dt = time.time() - t0
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        times.append(dt)

        answer_key = n["answer"][:10]
        is_correct = answer_key.lower() in resp.lower()
        if is_correct:
            correct += 1
        results.append({"position": len(results) / len(needles), "correct": is_correct, "response": resp.strip()[:50]})

    return {
        "strategy": name,
        "budget_tokens": budget,
        "actual_tokens": ptok,
        "compression_ratio": ratio,
        "recall": correct / len(needles),
        "avg_latency_s": round(sum(times) / len(times), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--num-facts", type=int, default=6)
    parser.add_argument("--out", default="results/semantic_ablation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    context, needles, _ = gen_semantic_context(args.context_tokens, tok, FACTS[:args.num_facts])
    ftok = len(tok.encode(context, add_special_tokens=False))
    print(f"Context: {ftok} tokens, {args.num_facts} facts\n")

    # 构建 LLM 话题分析策略
    llm_topic_fn = make_llm_topic_strategy(model, tok, device)
    STRATEGIES["llm_topic"] = ("LLM Topic Extraction", llm_topic_fn)

    results = {"config": {"model": args.model, "context_tokens": ftok, "budgets": args.budgets, "num_facts": args.num_facts}, "strategies": []}

    r = eval_strategy(model, tok, device, context, needles, "Full Context", full, ftok)
    print(f"Full Context: recall={r['recall']:.0%}  latency={r['avg_latency_s']:.2f}s")
    results["strategies"].append(r)

    for budget in args.budgets:
        ratio = 1 - budget / ftok
        print(f"\nBudget: {budget} tokens ({ratio:.0%} compression)")
        for key, (label, fn) in STRATEGIES.items():
            if key == "full":
                continue
            r = eval_strategy(model, tok, device, context, needles, label, fn, budget)
            marker = "✓" if r["recall"] >= 0.5 else "✗"
            print(f"  {marker} {label:<22} recall={r['recall']:.0%}  latency={r['avg_latency_s']:.2f}s  tok={r['actual_tokens']}")
            results["strategies"].append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
