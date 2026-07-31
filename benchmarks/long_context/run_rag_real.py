"""
真实 RAG 管道 benchmark：使用项目 services/rag/ 的真实组件。

管道：
  真实书籍文本 → RecursiveChunker 分块 → EmbeddingModel(bge-small-en) 建库
  → HybridRetriever(BM25 + Dense) 检索 → Qwen3-0.6B 基于检索结果回答

对比：RAG(top-k) vs 全量上下文 vs 截断，在真实文本 needle 任务上。

用法：
    .venv/bin/python benchmarks/long_context/run_rag_real.py
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目真实 RAG 组件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "services", "rag", "src"))
from chunker import RecursiveChunker
from embedding import EmbeddingModel
from vector_store import InMemoryVectorStore
from retriever import DenseRetriever, BM25Retriever, HybridRetriever

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_real_text(name: str = "pride_and_prejudice") -> str:
    path = os.path.join(DATA_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def prepare_context(text: str, target_tokens: int, tokenizer) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def insert_needles(context: str, num_needles: int, seed: int = 42) -> Tuple[str, List[Dict]]:
    """插入多对象 multikey needle：每个对象有独立 code，问题可定向检索。"""
    import random
    random.seed(seed)
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
             "Ivy", "Jack", "Kate", "Leo"]
    sentences = re.split(r'(?<=[.!?])\s+', context)
    positions = sorted([(i + 1) / (num_needles + 1) for i in range(num_needles)])
    out, needles = [], []
    prev = 0
    for i, pos in enumerate(positions):
        sidx = max(prev, min(int(pos * len(sentences)), len(sentences) - 1))
        name = names[i % len(names)]
        code = f"C{i:03d}"
        out.append(" ".join(sentences[prev:sidx]))
        out.append(f" {name}'s secret code is {code}. ")
        needles.append({"needle": code, "answer": code, "position": round(pos, 3),
                        "question": f"What is {name}'s secret code?"})
        prev = sidx
    out.append(" ".join(sentences[prev:]))
    return "".join(out), needles


def build_rag_index(text: str, chunk_size: int = 200) -> Dict:
    """使用项目真实 RAG 组件建索引。"""
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=20)
    chunks = chunker.chunk(text)
    if not chunks:
        return None

    embedder = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    store = InMemoryVectorStore(dimension=384)
    sparse = BM25Retriever()

    ids, vectors, metadatas, sparse_docs = [], [], [], []
    for i, chunk in enumerate(chunks):
        cid = f"chunk_{i}"
        ids.append(cid)
        vectors.append(embedder.embed(chunk))
        metadatas.append({"text": chunk, "chunk_index": i})
        sparse_docs.append({"id": cid, "text": chunk})

    store.add_batch(ids, vectors, metadatas)
    sparse.index(sparse_docs)

    dense = DenseRetriever(store, embedder)
    hybrid = HybridRetriever(dense, sparse, dense_weight=0.5)
    return {"hybrid": hybrid, "chunks": chunks}


def retrieve_context(rag_index: Dict, question: str, top_k: int = 3) -> str:
    """混合检索 top-k chunk 拼接为上下文。"""
    results = rag_index["hybrid"].retrieve(question, k=top_k)
    texts = [r.get("text", r.get("metadata", {}).get("text", "")) for r in results]
    return " ".join(t for t in texts if t)


def eval_with_retrieval(model, tok, device, rag_index, needles, top_k, tokenizer):
    """用 RAG 检索的上下文回答问题。"""
    correct = 0
    results = []
    times = []

    for n in needles:
        q = n["question"]
        context = retrieve_context(rag_index, q, top_k=top_k)
        prompt = context + "\n\n" + q
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        dt = time.time() - t0
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        times.append(dt)

        is_correct = n["answer"] in resp
        if is_correct:
            correct += 1
        results.append({"position": n["position"], "correct": is_correct, "response": resp.strip()[:40]})

    return {
        "strategy": f"RAG (top-{top_k})",
        "recall": correct / len(needles),
        "avg_latency_s": round(sum(times) / len(times), 3),
        "results": results,
    }


def eval_full(model, tok, device, context, needles):
    """全量上下文 baseline。"""
    correct = 0
    results = []
    times = []
    for n in needles:
        q = n["question"]
        prompt = context + "\n\n" + q
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        dt = time.time() - t0
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        times.append(dt)
        is_correct = n["answer"] in resp
        if is_correct:
            correct += 1
        results.append({"position": n["position"], "correct": is_correct, "response": resp.strip()[:40]})
    return {
        "strategy": "Full Context",
        "recall": correct / len(needles),
        "avg_latency_s": round(sum(times) / len(times), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--num-needles", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--out", default="results/rag_real.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Book: {args.book}, Model: {args.model}")
    book_text = load_real_text(args.book)
    context = prepare_context(book_text, args.context_tokens, tok)
    context_with_needles, needles = insert_needles(context, args.num_needles)
    ftok = len(tok.encode(context_with_needles, add_special_tokens=False))
    print(f"Context: {ftok} tokens, {args.num_needles} needles\n")

    print("Building real RAG index (chunker + bge embedding + hybrid retriever)...")
    rag_index = build_rag_index(context_with_needles, chunk_size=args.chunk_size)
    print(f"Indexed {len(rag_index['chunks'])} chunks\n")

    results = {"config": {"model": args.model, "book": args.book, "context_tokens": ftok, "num_needles": args.num_needles, "chunk_size": args.chunk_size, "top_k": args.top_k}, "strategies": []}

    # Full context baseline
    r_full = eval_full(model, tok, device, context_with_needles, needles)
    print(f"Full Context: recall={r_full['recall']:.0%}  latency={r_full['avg_latency_s']:.2f}s")
    results["strategies"].append(r_full)

    # RAG with different top-k
    for k in args.top_k:
        r_rag = eval_with_retrieval(model, tok, device, rag_index, needles, k, tok)
        print(f"RAG (top-{k}):     recall={r_rag['recall']:.0%}  latency={r_rag['avg_latency_s']:.2f}s")
        results["strategies"].append(r_rag)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
