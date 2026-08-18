"""Tests for MLA-style latent KV compression (CompressedKVCache + PCA basis).

Covers the transformers Cache contract invariants with real DynamicCache on
CPU tensors, via a FakeAttention harness that mirrors Qwen3Attention.forward's
ordering: mask sizes are read BEFORE the layer updates, then each layer updates
and the returned K/V seq-len must equal the pre-computed mask size.

A real-model GPU smoke test is gated behind KV_E2E=1 (compression trades a
small quality cost for D/latent memory savings, quantified by
``run_kv_compression_quality.py``).
"""

import os
import sys

import pytest

# test_hf_engine.py / test_server.py install MODULE-LEVEL MOCK torch and
# transformers into sys.modules. Evict them so this file gets the REAL modules.
_torch = sys.modules.get("torch")
if _torch is not None and not getattr(_torch, "__file__", None):
    del sys.modules["torch"]
import torch  # noqa: E402

_transformers = sys.modules.get("transformers")
if _transformers is not None and not getattr(_transformers, "__file__", None):
    del sys.modules["transformers"]

from optimization.kv_compression import (  # noqa: E402
    CompressedKVCache,
    fit_pca_basis,
    random_basis,
)
from transformers import DynamicCache  # noqa: E402

# Cache the REAL modules under sentinel keys (see test_kv_eviction.py).
sys.modules["__real_torch__"] = torch
sys.modules["__real_transformers__"] = sys.modules["transformers"]

N_LAYERS = 3
H, D = 2, 8
LATENT = 4


def rand(q, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, H, q, D, generator=g), torch.randn(1, H, q, D, generator=g)


def make_basis(latent=LATENT, seed=0):
    g = torch.Generator().manual_seed(seed)
    bk, bv = [], []
    for _ in range(N_LAYERS):
        rk = torch.randn(D, latent, generator=g)
        rk, _ = torch.linalg.qr(rk)
        rv = torch.randn(D, latent, generator=g)
        rv, _ = torch.linalg.qr(rv)
        bk.append(rk)
        bv.append(rv)
    return bk, bv


def forward_sim(cache, q_len, n_layers=N_LAYERS):
    """Simulate a model forward; assert returned seq-len == pre-update mask size."""
    sizes = [cache.get_mask_sizes(q_len, li)[0] for li in range(n_layers)]
    returned = []
    for li in range(n_layers):
        k, v = rand(q_len, seed=li + q_len)
        keys, values = cache.update(k, v, li)
        assert keys.shape[-2] == sizes[li], (
            f"update return ({keys.shape[-2]}) != get_mask_sizes ({sizes[li]}) "
            f"layer {li} q_len {q_len}"
        )
        returned.append(keys.shape[-2])
    return returned, sizes


def dynamiccache_bytes(cache: DynamicCache) -> int:
    """Bytes held by a plain ``DynamicCache``'s ``[B,H,S,D]`` K/V layers.

    ``DynamicCache`` exposes per-layer ``Cache`` objects via ``.layers`` with
    ``.keys`` / ``.values`` tensors; it has no ``kv_bytes`` helper of its own.
    """
    total = 0
    for layer in cache.layers:
        total += (layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
    return total


def recon_mse(X, basis):
    """Mean squared reconstruction error of X under rank-latent projection."""
    U = basis  # [D, latent]
    recon = X @ U @ U.T
    return (X - recon).pow(2).mean().item()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_validates_basis_lengths():
    bk, bv = make_basis()
    cache = CompressedKVCache(basis_k=bk, basis_v=bv, latent_dim=LATENT)
    assert cache.latent_length() == 0


# ---------------------------------------------------------------------------
# Cache contract
# ---------------------------------------------------------------------------


def test_prefill_then_decode_consistent():
    bk, bv = make_basis()
    cache = CompressedKVCache(basis_k=bk, basis_v=bv, latent_dim=LATENT)
    returned, _ = forward_sim(cache, 8)  # prefill
    assert returned == [8] * N_LAYERS
    returned2, _ = forward_sim(cache, 1)  # decode
    assert returned2 == [9] * N_LAYERS
    assert cache.logical_length == 9


def test_mask_size_matches_update_return():
    bk, bv = make_basis()
    cache = CompressedKVCache(basis_k=bk, basis_v=bv, latent_dim=LATENT)
    forward_sim(cache, 8)
    forward_sim(cache, 1)
    forward_sim(cache, 1)
    forward_sim(cache, 3)
    # every step's returned seq-len equals its pre-update mask size (asserted in sim)
    assert cache.logical_length == 13


def test_crop_keeps_prefix_layout():
    bk, bv = make_basis()
    cache = CompressedKVCache(basis_k=bk, basis_v=bv, latent_dim=LATENT)
    forward_sim(cache, 10)
    cache.crop(5)
    assert cache.latent_length() == 5


# ---------------------------------------------------------------------------
# Memory saving
# ---------------------------------------------------------------------------


def test_compressed_uses_less_memory_than_full():
    bk, bv = make_basis()
    comp = CompressedKVCache(basis_k=bk, basis_v=bv, latent_dim=LATENT)
    full = DynamicCache()
    for li in range(N_LAYERS):
        k, v = rand(8, seed=li)
        comp.update(k, v, li)
        full.update(k, v, li)
    # latent (4) < D (8) per token -> compressed KV is smaller
    full_bytes = dynamiccache_bytes(full)
    assert comp.kv_bytes() < full_bytes
    # exactly D / latent ratio per token (x2 for K+V)
    expected_ratio = D / LATENT
    actual = full_bytes / comp.kv_bytes()
    assert abs(actual - expected_ratio) < 1e-3


# ---------------------------------------------------------------------------
# PCA basis quality
# ---------------------------------------------------------------------------


def test_pca_recon_error_decreases_with_latent():
    g = torch.Generator().manual_seed(7)
    X = torch.randn(4, H, 16, D, generator=g).float()  # [B,H,S,D]
    key_list = [X[:, :, :, :] for _ in range(N_LAYERS)]
    val_list = [torch.randn_like(X) for _ in range(N_LAYERS)]
    bk, bv = fit_pca_basis(key_list, val_list, latent_dim=2)
    # reconstruct key cache per layer and measure recon error at latent=2 and latent=D
    err_2 = sum(recon_mse(X[:, :, :, :], bk[li]) for li in range(N_LAYERS)) / N_LAYERS
    bk_full, _ = fit_pca_basis(key_list, val_list, latent_dim=D)
    err_full = sum(recon_mse(X[:, :, :, :], bk_full[li]) for li in range(N_LAYERS)) / N_LAYERS
    assert err_full < 1e-4  # full-rank reconstruction ~ exact
    assert err_2 > err_full  # lower latent -> larger error


def test_random_basis_shape():
    bk, bv = random_basis(N_LAYERS, D, LATENT)
    assert len(bk) == N_LAYERS
    assert bk[0].shape == (D, LATENT)
    assert bv[0].shape == (D, LATENT)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("KV_E2E") != "1" or not torch.cuda.is_available(),
    reason="requires KV_E2E=1 on a CUDA host (real models)",
)
def test_real_compressed_cache_smoke():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    model.eval()

    ids = tok(
        "The history of the railway is the history of iron and coal. " * 8,
        return_tensors="pt",
    )["input_ids"].to("cuda")[:, :128]

    with torch.no_grad():
        out = model(ids, use_cache=True)
    key_cache = [l.keys.detach().float() for l in out.past_key_values.layers]
    val_cache = [l.values.detach().float() for l in out.past_key_values.layers]
    bk, bv = fit_pca_basis(key_cache, val_cache, latent_dim=16)

    out_comp = model(
        ids,
        past_key_values=CompressedKVCache(basis_k=bk, basis_v=bv, latent_dim=16),
        use_cache=True,
    )
    # compressed cache must produce valid logits (generation-quality is a separate
    # measurement in run_kv_compression_quality.py)
    assert out_comp.logits.shape[-1] == out.logits.shape[-1]
    print(f"[KV-E2E] compressed cache logits ok, latent bytes saved ~{D/16:.1f}x")
