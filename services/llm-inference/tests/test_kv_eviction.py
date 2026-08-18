"""Tests for the real StreamingLLM-style KV eviction (SinkWindowCache).

Covers the transformers Cache contract invariants with real DynamicLayer on
CPU tensors, via a FakeAttention harness that mimics Qwen3Attention.forward's
exact ordering: mask sizes are read BEFORE the layer updates, then each layer
updates and the returned K/V seq-len must equal the pre-computed mask size.
A real-model GPU smoke test is gated behind KV_E2E=1.
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

# src/ is on the path via pyproject.toml pythonpath=["src"]
from optimization.kv_eviction import (  # noqa: E402
    SinkWindowCache,
    evict_to_sink_window,
)
from transformers import DynamicCache  # noqa: E402

# Cache the REAL modules under sentinel keys. test_server.py runs AFTER this
# file alphabetically and replaces sys.modules["torch"]/["transformers"] with
# mocks; test_speculative_decoding.py then restores from these caches instead
# of re-importing (re-importing double-registers torch's C extensions and
# raises "Only a single TORCH_LIBRARY...").
sys.modules["__real_torch__"] = torch
sys.modules["__real_transformers__"] = sys.modules["transformers"]

N_LAYERS = 3
H, D = 2, 4


def rand(q, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, H, q, D, generator=g), torch.randn(1, H, q, D, generator=g)


def forward_sim(cache, q_len, n_layers=N_LAYERS):
    """Simulate a model forward with the real Qwen3 ordering.

    Returns (returned_seqs, precomputed_mask_sizes). The core invariant
    (returned K/V seq-len == get_mask_sizes, read on the pre-update state) is
    asserted inside.
    """
    sizes = [cache.get_mask_sizes(q_len, li)[0] for li in range(n_layers)]
    returned = []
    for li in range(n_layers):
        k, v = rand(q_len)
        keys, values = cache.update(k, v, li)
        assert keys.shape[-2] == sizes[li], (
            f"update return ({keys.shape[-2]}) != get_mask_sizes ({sizes[li]}) "
            f"layer {li} q_len {q_len}"
        )
        returned.append(keys.shape[-2])
    return returned, sizes


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_validates_sizes():
    with pytest.raises(ValueError):
        SinkWindowCache(sink_size=-1, window_size=8)
    with pytest.raises(ValueError):
        SinkWindowCache(sink_size=0, window_size=0)


def test_get_max_length_returns_cap():
    cache = SinkWindowCache(sink_size=4, window_size=512)
    assert cache.get_max_length() == 516


# ---------------------------------------------------------------------------
# Logical length tracking
# ---------------------------------------------------------------------------


def test_logical_length_monotonic_prefill_then_decode():
    cache = SinkWindowCache(sink_size=2, window_size=3)  # cap=5
    forward_sim(cache, 8)
    assert cache.logical_length == 8
    for _ in range(3):
        forward_sim(cache, 1)
    assert cache.logical_length == 11  # 8 + 3, monotonic
    # never decreases even though physical is capped at 5
    assert cache._physical_len(0) == 5


def test_get_seq_length_is_logical_not_physical():
    cache = SinkWindowCache(sink_size=2, window_size=3)
    forward_sim(cache, 8)
    forward_sim(cache, 1)  # evict to cap=5
    assert cache.get_seq_length() == 9  # logical
    assert cache._physical_len(0) == 5  # physical


def test_get_query_offset_equals_seq_length():
    cache = SinkWindowCache(sink_size=2, window_size=3)
    forward_sim(cache, 8)
    forward_sim(cache, 1)
    assert cache.get_query_offset() == cache.get_seq_length() == 9


# ---------------------------------------------------------------------------
# Eviction behavior
# ---------------------------------------------------------------------------


def test_no_eviction_during_prefill():
    cache = SinkWindowCache(sink_size=2, window_size=3)  # cap=5
    returned, _ = forward_sim(cache, 8)  # prefill 8 > cap but not evicted
    assert returned == [8] * N_LAYERS
    assert cache._physical_len(0) == 8


def test_physical_capped_on_decode():
    cache = SinkWindowCache(sink_size=2, window_size=3)  # cap=5
    forward_sim(cache, 8)
    forward_sim(cache, 1)
    for li in range(N_LAYERS):
        assert cache.layers[li].keys.shape[-2] == 5


def test_first_decode_step_evicts_middle():
    cache = SinkWindowCache(sink_size=2, window_size=3)  # cap=5
    forward_sim(cache, 8)  # full context
    returned, sizes = forward_sim(cache, 1)  # first decode step
    assert returned == [5] * N_LAYERS
    assert sizes == [5] * N_LAYERS
    assert cache.logical_length == 9
    assert cache._physical_len(0) == 5


def test_keeps_sink_and_window_tokens():
    cache = SinkWindowCache(sink_size=2, window_size=3)  # cap=5
    C = 10
    # position-marker key: value at position p == p
    k = torch.zeros(1, 1, C, 1)
    k[0, 0, :, 0] = torch.arange(C).float()
    v = torch.zeros(1, 1, C, 1)
    cache.update(k, v, 0)  # prefill layer 0
    k_dec = torch.zeros(1, 1, 1, 1)
    k_dec[0, 0, 0, 0] = 10.0  # decode token at logical position 10
    cache.update(k_dec, torch.zeros(1, 1, 1, 1), 0)  # evict -> [sink|window]
    kept = cache.layers[0].keys[0, 0, :, 0]
    expected = torch.cat([torch.arange(2), torch.arange(8, 11)]).float()
    assert torch.equal(kept, expected)


def test_single_token_prefill_noop_when_under_cap():
    cache = SinkWindowCache(sink_size=2, window_size=3)  # cap=5
    returned, _ = forward_sim(cache, 1)  # 1-token "prefill", under cap
    assert returned == [1] * N_LAYERS
    assert cache._physical_len(0) == 1  # no duplication, no truncation


def test_get_mask_sizes_matches_update_return():
    cache = SinkWindowCache(sink_size=2, window_size=3)
    # combos: first prefill, decode past cap, prefill-after-decode
    forward_sim(cache, 8)  # (8,0) -> returns 8
    forward_sim(cache, 1)  # capped -> returns 5
    forward_sim(cache, 1)  # capped -> returns 5
    forward_sim(cache, 3)  # uncapped -> returns 8 (5+3)


def test_evict_to_sink_window_helper():
    dc = DynamicCache()
    for li in range(N_LAYERS):
        k, v = rand(8)
        dc.update(k, v, li)
    evict_to_sink_window(dc, 2, 3)
    assert dc.layers[0].keys.shape[-2] == 5
    assert dc.layers[2].keys.shape[-2] == 5
    # idempotent when under cap
    evict_to_sink_window(dc, 2, 3)
    assert dc.layers[0].keys.shape[-2] == 5


def test_crop_raises():
    cache = SinkWindowCache(sink_size=2, window_size=3)
    forward_sim(cache, 8)
    with pytest.raises(NotImplementedError):
        cache.crop(5)


def test_kv_bytes_counts_tensors():
    cache = SinkWindowCache(sink_size=2, window_size=3)
    forward_sim(cache, 8)  # 3 layers x [1,2,8,4] fp32 x 2 (K+V)
    expected = 3 * (1 * 2 * 8 * 4 * 4) * 2
    assert cache.kv_bytes() == expected


# ---------------------------------------------------------------------------
# Real-model GPU smoke (opt-in: KV_E2E=1)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("KV_E2E") != "1" or not torch.cuda.is_available(),
    reason="requires KV_E2E=1 on a CUDA host (real models)",
)
def test_real_sink_window_cache_smoke():
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
    )["input_ids"].to("cuda")
    ids = ids[:, :128]

    # 1) while under cap, SinkWindowCache plumbing must be a no-op: identical logits
    with torch.no_grad():
        o_base = model(ids, past_key_values=DynamicCache(), use_cache=True)
        o_evict = model(
            ids,
            past_key_values=SinkWindowCache(sink_size=4, window_size=512),
            use_cache=True,
        )
    assert torch.equal(o_base.logits, o_evict.logits), "logits differ before eviction"

    # 2) generation with eviction (window smaller than context) works
    out = model.generate(
        ids,
        max_new_tokens=8,
        do_sample=False,
        past_key_values=SinkWindowCache(sink_size=4, window_size=64),
    )
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    assert text.strip()
    print(f"[KV-E2E] ok: {text[:40]!r}")
