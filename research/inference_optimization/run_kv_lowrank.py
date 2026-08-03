"""
KV cache low-rank compression analysis (MLA-inspired, inference-side).

Goal: verify that real KV caches have strong low-rank structure,
supporting the MLA (Multi-head Latent Attention) argument that KV
can be compressed to a small latent per token.

Measurement (all real, no simulation):
1. Load Qwen3-0.6B, prefill real text (Project Gutenberg) to get KV cache
2. SVD each layer's K/V, measure singular value energy retention vs rank
3. Rank needed to retain 99% / 95% energy → implied KV memory savings
4. Truncated-SVD reconstruction error of attention output vs original

Usage: .venv/bin/python research/inference_optimization/run_kv_lowrank.py
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
BOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "long_context", "data")


def load_book_text(max_tokens_ctx: int = 2048) -> str:
    """Load real book text from long_context data (Project Gutenberg)."""
    candidates = [
        os.path.join(BOOK_DIR, "pride_and_prejudice.txt"),
        os.path.join(BOOK_DIR, "data", "pride_and_prejudice.txt"),
    ]
    # fix: BOOK_DIR already points to data dir
    candidates.insert(0, os.path.join(BOOK_DIR, "pride_and_prejudice.txt"))
    for c in candidates:
        if os.path.exists(c):
            with open(c, encoding="utf-8", errors="ignore") as f:
                return f.read()[: 4 * max_tokens_ctx]
    # Fallback: generate deterministic pseudo-text is NOT allowed (real data only)
    raise FileNotFoundError(f"No book text found in {BOOK_DIR}")


def get_kv_cache(model, tokenizer, device, text: str, max_tokens: int) -> List[torch.Tensor]:
    """Prefill text and return key cache (list of [B,H,S,D] per layer)."""
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][:, :max_tokens].to(device)
    with torch.no_grad():
        # use_cache=True keeps DynamicCache; return key cache via outputs
        out = model(tokens, use_cache=True)
    # New transformers DynamicCache uses .layers[i].keys/.values
    if hasattr(out.past_key_values, "layers"):
        key_cache = [l.keys.detach().float() for l in out.past_key_values.layers]
        value_cache = [l.values.detach().float() for l in out.past_key_values.layers]
    else:  # legacy
        key_cache = [k.detach().float() for k in out.past_key_values.key_cache]
        value_cache = [v.detach().float() for v in out.past_key_values.value_cache]
    return key_cache, value_cache


def svd_energy_all_heads(tensor: torch.Tensor) -> Dict:
    """SVD energy across ALL heads: rank needed for energy retention.

    This is SEQUENCE-dimension low-rank (token redundancy).
    """
    B, H, S, D = tensor.shape
    ranks_95, ranks_99, ranks_999 = [], [], []
    for b in range(B):
        for h in range(H):
            head = tensor[b, h]
            U, Svals, Vt = torch.linalg.svd(head, full_matrices=False)
            total_energy = (Svals**2).sum()
            cum = torch.cumsum(Svals**2, dim=0) / total_energy
            ranks_99.append(int((cum >= 0.99).nonzero()[0].item()) + 1)
            ranks_95.append(int((cum >= 0.95).nonzero()[0].item()) + 1)
            if (cum >= 0.999).any():
                ranks_999.append(int((cum >= 0.999).nonzero()[0].item()) + 1)
            else:
                ranks_999.append(D)
    return {
        "seq_len": S, "head_dim": D, "n_heads": H,
        "rank95_avg": round(sum(ranks_95) / len(ranks_95), 1),
        "rank99_avg": round(sum(ranks_99) / len(ranks_99), 1),
        "rank999_avg": round(sum(ranks_999) / len(ranks_999), 1),
        "rank99_min": min(ranks_99), "rank99_max": max(ranks_99),
    }


def dim_redundancy_mla(tensor: torch.Tensor) -> Dict:
    """DIMENSION-dimension low-rank: the MLA analogy.

    MLA compresses each token's K/V to a low-dim latent. The inference-side
    analog: the D-dim K vectors across S tokens live in a low-dim subspace.
    We compute the DxD covariance (Gram) spectrum — principal components of
    the K representation. If a few PCs hold most energy, per-token K can be
    represented with fewer dims (latent compression).
    """
    B, H, S, D = tensor.shape
    pc_ratios = []  # energy fraction in top PCs per head
    for b in range(B):
        for h in range(H):
            K = tensor[b, h]  # [S, D]
            Kc = K - K.mean(dim=0, keepdim=True)  # center
            gram = Kc.T @ Kc / S  # [D, D]
            evals = torch.linalg.eigvalsh(gram).flip(0)  # descending
            evals = evals.clamp(min=0)
            total = evals.sum()
            if total < 1e-12:
                pc_ratios.append(1.0)
                continue
            cum = torch.cumsum(evals, dim=0) / total
            r90 = int((cum >= 0.90).nonzero()[0].item()) + 1
            r95 = int((cum >= 0.95).nonzero()[0].item()) + 1
            r99 = int((cum >= 0.99).nonzero()[0].item()) + 1
            pc_ratios.append((r90, r95, r99))
    r90 = sum(pc[0] for pc in pc_ratios) / len(pc_ratios)
    r95 = sum(pc[1] for pc in pc_ratios) / len(pc_ratios)
    r99 = sum(pc[2] for pc in pc_ratios) / len(pc_ratios)
    return {
        "head_dim": D,
        "pc_90_avg": round(r90, 1),
        "pc_95_avg": round(r95, 1),
        "pc_99_avg": round(r99, 1),
        "implied_latent_save_95": round(1 - r95 / D, 4),
        "implied_latent_save_99": round(1 - r99 / D, 4),
    }


def truncated_recon_error(kv_list: List[torch.Tensor], rank: int) -> float:
    """Reconstruction error of truncated-SVD K/V vs original (all layers)."""
    errs = []
    for kv in kv_list:
        B, H, S, D = kv.shape
        for b in range(B):
            for h in range(H):
                M = kv[b, h]
                U, Svals, Vt = torch.linalg.svd(M, full_matrices=False)
                M_approx = (U[:, :rank] * Svals[:rank]) @ Vt[:rank, :]
                errs.append(F.mse_loss(M_approx, M).item())
    return sum(errs) / len(errs)


def attention_output_error(key, value, rank: int, head_dim: int) -> float:
    """Compare attention output with original vs truncated-SVD K/V."""
    B, H, S, D = key.shape
    # Random query (deterministic) — measures structural error, not semantics
    torch.manual_seed(0)
    q = torch.randn(B, H, 1, D, device=key.device) / (D**0.5)

    errs = []
    for b in range(B):
        for h in range(H):
            K, V = key[b, h], value[b, h]
            Uk, Sk, Vtk = torch.linalg.svd(K, full_matrices=False)
            Uv, Sv, Vtv = torch.linalg.svd(V, full_matrices=False)
            K_approx = (Uk[:, :rank] * Sk[:rank]) @ Vtk[:rank, :]
            V_approx = (Uv[:, :rank] * Sv[:rank]) @ Vtv[:rank, :]

            attn_orig = F.softmax(q[b, h] @ K.T / (head_dim**0.5), dim=-1)
            attn_approx = F.softmax(q[b, h] @ K_approx.T / (head_dim**0.5), dim=-1)
            out_orig = attn_orig @ V
            out_approx = attn_approx @ V_approx
            rel_err = F.mse_loss(out_approx, out_orig).item() / (out_orig.pow(2).mean().item() + 1e-8)
            errs.append(rel_err)
    return sum(errs) / len(errs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--output", default="results/kv_lowrank.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")

    text = load_book_text(args.max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    model.eval()

    key_cache, value_cache = get_kv_cache(model, tokenizer, device, text, args.max_tokens)
    n_layers = len(key_cache)
    print(f"KV cache captured: {n_layers} layers, shape {list(key_cache[0].shape)}\n")

    # 1. Sequence-dimension low-rank (token redundancy) — all heads
    layer_idx = [0, n_layers // 2, n_layers - 1]
    energy = {}
    for li in layer_idx:
        e = svd_energy_all_heads(key_cache[li])
        energy[f"layer{li}"] = e
        print(f"  Layer {li}: rank95_avg={e['rank95_avg']} rank99_avg={e['rank99_avg']} "
              f"rank999_avg={e['rank999_avg']} (D={e['head_dim']})")

    # 1b. DIMENSION-dimension low-rank — the MLA analogy (latent compression)
    print("\n  Dimension-level redundancy (MLA analogy): per-token K dims")
    dim_energy = {}
    for li in layer_idx:
        d = dim_redundancy_mla(key_cache[li])
        dim_energy[f"layer{li}"] = d
        print(f"  Layer {li}: PC90={d['pc_90_avg']} PC95={d['pc_95_avg']} PC99={d['pc_99_avg']} "
              f"(of D={d['head_dim']}); latent save@95%={d['implied_latent_save_95']:.0%} "
              f"save@99%={d['implied_latent_save_99']:.0%}")

    # 2. Reconstruction error across all layers at each rank
    recon_errs = {}
    for r in args.ranks:
        k_err = truncated_recon_error(key_cache, r)
        v_err = truncated_recon_error(value_cache, r)
        recon_errs[str(r)] = {"k_mse": round(k_err, 6), "v_mse": round(v_err, 6)}
        print(f"  rank={r:>3}: K MSE={k_err:.2e}  V MSE={v_err:.2e}")

    # 3. Attention output relative error (layer 0) at each rank
    attn_errs = {}
    for r in args.ranks:
        err = attention_output_error(key_cache[0], value_cache[0], r, key_cache[0].shape[-1])
        attn_errs[str(r)] = round(err, 6)
        print(f"  rank={r:>3}: attention output rel err={err:.2e}")

    results = {
        "config": {"model": args.model, "max_tokens": args.max_tokens, "n_layers": n_layers},
        "token_redundancy_svd": energy,          # sequence-dim low rank
        "mla_analogy_dim_pca": dim_energy,        # dimension-dim low rank
        "recon_error": recon_errs,
        "attn_output_error": attn_errs,
        "interpretation": (
            "SVD rank (token_redundancy) reveals token-level redundancy — "
            "supports token pruning (H2O/StreamingLLM style). "
            "PCA dims (mla_analogy_dim_pca) reveal per-token K/V dimension "
            "redundancy — the inference-side analog of MLA latent compression. "
            "Both measured on real KV caches."
        ),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
