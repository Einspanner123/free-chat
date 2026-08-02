"""
LongBench passage_retrieval_en + 真实 RAG 检索

用项目真实 RAG 组件（BM25Retriever / DenseRetriever / HybridRetriever）
精确检索 top-K 段落，测端到端准确率。

对比：
- 关键词命中压缩（之前 74%）
- BM25 检索 top-k
- Dense(embedding) 检索 top-k
- Hybrid 检索 top-k

指标：段落号预测准确率

用法：.venv/bin/python benchmarks/longbench_v1/run_rag_retrieval.py
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

# 项目真实 RAG 组件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "services", "rag", "src"))
from retriever import BM25Retriever, DenseRetriever, HybridRetriever
from embedding import EmbeddingModel
from vector_store import InMemoryVectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "data")


def load_items() -> List[Dict]:
    with open(os.path.join(DATA_DIR, "passage_retrieval_en.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def split_paragraphs(context: str) -> List[Dict]:
    """把 context 切成段落，带段落号。"""
    paras = re.split(r'(?=Paragraph \d+:)', context)
    paras = [p for p in paras if p.strip()]
    docs = []
    for p in paras:
        m = re.match(r'Paragraph (\d+):', p)
        if m:
            num = m.group(1)
            text = p[m.end():].strip()
            docs.append({"id": f"P{num}", "num": num, "text": text})
    return docs


def build_retrievers(docs: List[Dict]):
    """构建 BM25 + Dense + Hybrid 检索器。"""
    # BM25
    bm25 = BM25Retriever()
    bm25.index([{"id": d["id"], "text": d["text"]} for d in docs])

    # Dense (bge embedding)
    embedder = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    store = InMemoryVectorStore(dimension=384)
    vectors = [embedder.embed(d["text"]) for d in docs]
    store.add_batch([d["id"] for d in docs], vectors, [{"text": d["text"]} for d in docs])
    dense = DenseRetriever(store, embedder)
    hybrid = HybridRetriever(dense, bm25, dense_weight=0.5)

    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}, embedder


def retrieve_topk(retrievers, docs, query, method: str, k: int) -> str:
    """用指定检索器取 top-k 段落拼接。"""
    if method == "bm25":
        results = retrievers["bm25"].retrieve(query, k=k)
    elif method == "dense":
        results = retrievers["dense"].retrieve(query, k=k)
    else:
        results = retrievers["hybrid"].retrieve(query, k=k)

    # 按 id 取回段落全文
    id_to_doc = {d["id"]: d for d in docs}
    selected = []
    for r in results:
        did = r["id"]
        if did in id_to_doc:
            d = id_to_doc[did]
            selected.append(f"Paragraph {d['num']}: {d['text']}")
    return "\n".join(selected)


def evaluate(model, tokenizer, device, items, method, k, max_new=10) -> Dict:
    correct = 0
    times = []
    per_item = []

    for item in items:
        docs = split_paragraphs(item["context"])
        retrievers, embedder = build_retrievers(docs)
        ctx = retrieve_topk(retrievers, docs, item["input"], method, k)
        used_tok = len(tokenizer.encode(ctx, add_special_tokens=False))

        prompt = f"Passages:\n{ctx}\n\nFind the passage that matches: {item['input']}\n\nAnswer with the paragraph number:"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = re.search(r'(\d+)', resp)
        pred_n = pred.group(1) if pred else None
        gold = re.search(r'(\d+)', item["answers"][0]).group(1)
        is_correct = pred_n == gold
        if is_correct:
            correct += 1
        per_item.append({
            "id": item.get("_id", ""), "gold": gold, "pred": pred_n, "correct": is_correct,
            "used_tokens": used_tok,
        })

    return {
        "method": method, "k": k,
        "accuracy": correct / len(items), "correct": correct, "total": len(items),
        "avg_latency_s": round(sum(times) / len(times), 2),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--output", default="results/rag_retrieval.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    items = load_items()[:args.samples]
    print(f"passage_retrieval_en: {len(items)} samples\n")

    results = {"config": {"model": args.model, "samples": len(items), "methods": args.methods, "top_k": args.top_k}, "results": []}

    for method in args.methods:
        for k in args.top_k:
            r = evaluate(model, tokenizer, device, items, method, k)
            print(f"  {method:<8} top-{k}: acc={r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            results["results"].append(r)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
