"""Tests for the REAL SpeculativeDecoder (Leviathan et al. 2023).

Covers exact rejection-sampling math, the logits warpers (gated exactly like
``transformers`` ``generate()``), and deterministic single-round behavior with
tiny fake torch models. A real-model GPU smoke test is gated behind
``SPEC_E2E=1`` so it never runs during a plain ``pytest tests``.
"""

import os
import sys
from types import SimpleNamespace

import pytest

# test_hf_engine.py / test_server.py install MODULE-LEVEL MOCK torch and
# transformers into sys.modules. Pytest collects test files alphabetically, so
# by the time this file runs, the mocks (created via ``types.ModuleType``,
# which sets ``__file__`` to None) may already be in place. Restore the REAL
# modules: test_kv_eviction.py (earlier alphabetically) cached them under
# sentinel keys, so we can restore WITHOUT re-importing — re-importing torch
# double-registers its C extensions and raises "Only a single TORCH_LIBRARY...".
# If they were never imported yet (partial runs), drop the mock and import fresh.
_torch = sys.modules.get("torch")
if _torch is not None and not getattr(_torch, "__file__", None):
    cached = sys.modules.get("__real_torch__")
    if cached is not None:
        sys.modules["torch"] = cached
    else:
        del sys.modules["torch"]
import torch  # noqa: E402

_transformers = sys.modules.get("transformers")
if _transformers is not None and not getattr(_transformers, "__file__", None):
    cached = sys.modules.get("__real_transformers__")
    if cached is not None:
        sys.modules["transformers"] = cached
    else:
        del sys.modules["transformers"]

# src/ is on the path via pyproject.toml pythonpath=["src"]
from optimization.speculative_decoding import (  # noqa: E402
    SpeculativeDecoder,
    SpeculativeStats,
)


# ---------------------------------------------------------------------------
# Tiny fakes: no real model, no GPU
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal tokenizer: shared vocab + dummy encode (same ids both sides)."""

    def __init__(self, vocab=4, probe_ids=None):
        self._vocab = vocab
        self._probe_ids = probe_ids if probe_ids is not None else [1, 2]

    def __len__(self):
        return self._vocab

    def encode(self, s):
        return list(self._probe_ids)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class FakeCache:
    """Minimal stand-in for ``DynamicCache``: only length + crop()."""

    def __init__(self, length=0):
        self.length = length

    def crop(self, max_length):
        self.length = max_length


def _peaked_logits(rows, vocab):
    """[1, L, vocab] logits peaked at ``rows[i]`` (deterministic softmax)."""
    L = len(rows)
    t = torch.full((1, L, vocab), -30.0)
    for i, tok in enumerate(rows):
        t[0, i, tok] = 30.0
    return t


class FakeModel:
    """Fake HF model: __call__(input_ids, past_key_values=None, use_cache=True)."""

    def __init__(self, logits_fn, vocab=4):
        self.logits_fn = logits_fn
        self.vocab = vocab
        self.device = torch.device("cpu")
        self.forward_calls = []  # input lengths
        self.crop_calls = []  # max_length args

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        input_len = input_ids.shape[1]
        self.forward_calls.append(input_len)
        base = past_key_values.length if past_key_values is not None else 0
        cache = FakeCache(base + input_len)
        orig_crop = cache.crop

        def _crop(max_length):
            orig_crop(max_length)
            self.crop_calls.append(max_length)

        cache.crop = _crop
        logits = self.logits_fn(input_len, past_key_values is not None)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def make_decoder(draft_fn, target_fn, vocab=4, gamma=5, **kwargs):
    draft = FakeModel(draft_fn, vocab)
    target = FakeModel(target_fn, vocab)
    tok = FakeTokenizer(vocab)
    decoder = SpeculativeDecoder(
        draft_model=draft,
        draft_tokenizer=tok,
        target_model=target,
        target_tokenizer=tok,
        gamma=gamma,
        device=torch.device("cpu"),
        temperature=0.7,
        top_p=0.8,
        top_k=40,
        repetition_penalty=1.0,
        **kwargs,
    )
    return decoder, draft, target


# ---------------------------------------------------------------------------
# Rejection sampling (exact math, injected rng)
# ---------------------------------------------------------------------------


def test_rejection_sampling_all_accept_when_p_ge_q():
    assert SpeculativeDecoder.rejection_sampling([0.5, 0.5], [0.9, 0.9]) == 2


def test_rejection_sampling_q_zero_accepts():
    assert SpeculativeDecoder.rejection_sampling([0.0, 0.5], [0.0, 0.9]) == 2


def test_rejection_sampling_first_rejection():
    # p < q at position 0, rng always rejects -> return 0
    assert SpeculativeDecoder.rejection_sampling(
        [0.9, 0.9], [0.1, 0.1], rng=lambda: 1.0
    ) == 0


def test_rejection_sampling_never_rejects():
    # p < q but rng always accepts -> all accepted
    assert SpeculativeDecoder.rejection_sampling(
        [0.9, 0.9], [0.1, 0.1], rng=lambda: 0.0
    ) == 2


def test_rejection_sampling_mid_rejection():
    # position 0 accepted (p >= q), position 1 rejected -> return 1
    assert SpeculativeDecoder.rejection_sampling(
        [0.5, 0.5], [0.9, 0.1], rng=lambda: 1.0
    ) == 1


def test_rejection_sampling_empty():
    assert SpeculativeDecoder.rejection_sampling([], []) == 0


def test_rejection_sampling_p_eq_q_accepts():
    # p == q -> accept even if rng would reject
    assert SpeculativeDecoder.rejection_sampling(
        [0.5, 0.5], [0.5, 0.5], rng=lambda: 1.0
    ) == 2


def test_rejection_sampling_empirical_accept_rate():
    # each position accepts with prob p/q = 0.5 -> P(reject at 0) ~ 0.5
    import random

    random.seed(42)
    q = [0.8] * 200
    p = [0.4] * 200
    first_rejected = 0
    for _ in range(200):
        n = SpeculativeDecoder.rejection_sampling(q, p, rng=random.random)
        if n == 0:
            first_rejected += 1
    assert 0.35 < first_rejected / 200 < 0.65


# ---------------------------------------------------------------------------
# Logits warpers (gated exactly like transformers generate())
# ---------------------------------------------------------------------------


def _warped_decoder(**overrides):
    d = object.__new__(SpeculativeDecoder)
    d.temperature = overrides.get("temperature", 1.0)
    d.top_p = overrides.get("top_p", 1.0)
    d.top_k = overrides.get("top_k", 0)
    d.repetition_penalty = overrides.get("repetition_penalty", 1.0)
    return d


def test_warp_temperature():
    d = _warped_decoder(temperature=2.0)
    out = d._warp_logits(torch.tensor([1.0, 2.0, 3.0]), [])
    assert torch.allclose(out, torch.tensor([0.5, 1.0, 1.5]))


def test_warp_top_k_keeps_exactly_k():
    d = _warped_decoder(top_k=2)
    logits = torch.tensor([0.1, 5.0, 3.0, -1.0])
    out = d._warp_logits(logits, [])
    assert (out != float("-inf")).sum().item() == 2
    assert out[1] != float("-inf") and out[2] != float("-inf")
    assert out[0] == float("-inf") and out[3] == float("-inf")


def test_warp_top_p_keeps_mass():
    d = _warped_decoder(top_p=0.5)
    out = d._warp_logits(torch.tensor([0.0, 10.0, 0.0, 0.0]), [])
    probs = torch.softmax(out, dim=-1)
    kept = probs[probs > 0]
    assert kept.sum().item() >= 0.5  # kept mass covers top_p
    assert probs[1].item() > 0.99  # dominant token preserved
    assert (probs > 0).sum().item() >= 1


def test_warp_repetition_penalty():
    d = _warped_decoder(repetition_penalty=2.0)
    logits = torch.tensor([0.5, -0.5, 3.0])
    out = d._warp_logits(logits, [0, 1])  # tokens 0 and 1 in context
    assert out[0].item() == pytest.approx(0.5 / 2.0)  # positive -> divided
    assert out[1].item() == pytest.approx(-0.5 * 2.0)  # negative -> multiplied
    assert out[2].item() == pytest.approx(3.0)  # not in context -> unchanged


def test_warp_gates_are_noops():
    d = _warped_decoder()  # temp=1.0, top_p=1.0, top_k=0, rep=1.0
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(d._warp_logits(logits.clone(), [0, 1]), logits)


def test_warp_softmax_normalizes():
    d = _warped_decoder(temperature=0.7, top_p=0.8, top_k=4)
    out = d._warp_logits(torch.randn(8), [1, 2, 3])
    assert torch.softmax(out, dim=-1).sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_init_rejects_temperature_zero():
    tok = FakeTokenizer(4)
    m = FakeModel(lambda L, hp: _peaked_logits([1] * L, 4), 4)
    with pytest.raises(ValueError):
        SpeculativeDecoder(
            draft_model=m, draft_tokenizer=tok, target_model=m,
            target_tokenizer=tok, temperature=0,
        )


def test_init_rejects_probe_mismatch():
    # Different token ids for real text -> hard error (e.g. Llama vs Qwen).
    draft_tok = FakeTokenizer(4, probe_ids=[1, 2])
    target_tok = FakeTokenizer(4, probe_ids=[3, 4])
    m = FakeModel(lambda L, hp: _peaked_logits([1] * L, 4), 4)
    with pytest.raises(ValueError):
        SpeculativeDecoder(
            draft_model=m, draft_tokenizer=draft_tok, target_model=m,
            target_tokenizer=target_tok,
        )


def test_init_allows_vocab_tail_mismatch():
    # Same probe ids but different len(tokenizer) -> OK (added-token tail,
    # e.g. Qwen2.5 draft for Qwen3 target).
    draft_tok = FakeTokenizer(4, probe_ids=[1, 2])
    target_tok = FakeTokenizer(8, probe_ids=[1, 2])
    m = FakeModel(lambda L, hp: _peaked_logits([1] * L, 4), 4)
    decoder = SpeculativeDecoder(
        draft_model=m, draft_tokenizer=draft_tok, target_model=m,
        target_tokenizer=target_tok,
    )
    assert decoder is not None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_acceptance_rate():
    s = SpeculativeStats(n_draft_accepted=4, n_draft_proposed=5)
    assert s.acceptance_rate == 0.8
    assert SpeculativeStats().acceptance_rate == 0.0


def test_stats_expected_tokens_per_verify():
    s = SpeculativeStats(n_draft_accepted=4, n_draft_proposed=5)  # ar=0.8
    assert s.expected_tokens_per_verify(5) == pytest.approx((1 - 0.8 ** 6) / 0.2)
    s_full = SpeculativeStats(n_draft_accepted=5, n_draft_proposed=5)
    assert s_full.expected_tokens_per_verify(5) == 6.0


# ---------------------------------------------------------------------------
# Deterministic single-round behavior (tiny fake models)
# ---------------------------------------------------------------------------


def test_round_all_accepted():
    # Draft always proposes token 1; target agrees -> one verify accepts all 5.
    draft_fn = lambda L, hp: _peaked_logits([1] * L, 4)
    target_fn = lambda L, hp: _peaked_logits([1] * L, 4)
    decoder, draft, target = make_decoder(draft_fn, target_fn)
    prompt = torch.tensor([[0, 0, 0]])  # prompt length 3

    ids, stats = decoder.generate(prompt, max_tokens=5, eos_token_id=None)

    assert ids == [1, 1, 1, 1, 1]
    assert stats.n_draft_proposed == 5
    assert stats.n_draft_accepted == 5
    assert stats.acceptance_rate == 1.0
    # draft: prefill(prompt) + 5 single-token propose steps (KV reuse)
    assert draft.forward_calls == [3, 1, 1, 1, 1, 1]
    # target: prefill(prompt) + 1 parallel verify of the whole candidate
    assert target.forward_calls == [3, 5]
    # all accepted -> no KV rollback
    assert target.crop_calls == []


def test_round_rejection_correction():
    # Draft proposes token 1; target peaks at token 3 -> p~=0 for cand,
    # rejection at position 0, correction token sampled from target = 3.
    draft_fn = lambda L, hp: _peaked_logits([1] * L, 4)
    target_fn = lambda L, hp: _peaked_logits([3] * L, 4)
    decoder, draft, target = make_decoder(draft_fn, target_fn)
    prompt = torch.tensor([[0, 0, 0]])

    ids, stats = decoder.generate(prompt, max_tokens=1, eos_token_id=None)

    assert ids == [3]  # correction token
    assert stats.n_draft_accepted == 0
    assert stats.n_draft_proposed == 5
    # both KV caches rolled back to prompt_len(3) + accepted(0)
    assert target.crop_calls == [3]
    assert draft.crop_calls == [3]
    # target: prefill(3) + verify(5) + correction re-forward(1)
    assert target.forward_calls == [3, 5, 1]
    # draft: prefill(3) + 5 propose steps + correction re-forward(1)
    assert draft.forward_calls == [3, 1, 1, 1, 1, 1, 1]


def test_round_stops_on_eos():
    draft_fn = lambda L, hp: _peaked_logits([1] * L, 4)
    target_fn = lambda L, hp: _peaked_logits([1] * L, 4)
    decoder, _, _ = make_decoder(draft_fn, target_fn)
    prompt = torch.tensor([[0, 0, 0]])

    ids, stats = decoder.generate(prompt, max_tokens=10, eos_token_id=1)

    assert ids == [1]  # stops at the first EOS token
    assert stats.acceptance_rate == 1.0


def test_target_verify_logit_alignment():
    # p[0] must come from the prefill's LAST-position logits (predicts cand[0]);
    # p[i>=1] from verify logits row i-1 (predicts cand[i]).
    draft_fn = lambda L, hp: _peaked_logits([1] * L, 4)

    def target_fn(L, has_past):
        if not has_past:
            return _peaked_logits([1] * L, 4)  # prefill: t_first predicts token 1
        return _peaked_logits([2, 1, 0][:L], 4)  # verify rows peak at 2,1,0

    decoder, _, target = make_decoder(draft_fn, target_fn)
    prompt = torch.tensor([[0, 0, 0]])
    cand = torch.tensor([1, 1, 1])  # 1-D, as produced by the decoder

    prefill_out = target(prompt, past_key_values=None, use_cache=True)
    t_first = prefill_out.logits[0, -1]
    p, _, _ = decoder._target_verify(
        cand, prefill_out.past_key_values, t_first, prompt[0].tolist(), cur_len=3
    )

    # p[0]: P(cand[0]=1 | prompt)      <- prefill last position (peaked at 1)  ~1
    # p[1]: P(cand[1]=1 | prompt,c0)   <- verify logits[0] (peaked at 2)        ~0
    # p[2]: P(cand[2]=1 | prompt,c0,c1)<- verify logits[1] (peaked at 1)        ~1
    assert p[0] > 0.99
    assert p[1] < 0.01
    assert p[2] > 0.99


# ---------------------------------------------------------------------------
# Expected speedup formula
# ---------------------------------------------------------------------------


def test_estimate_speedup_formula():
    d = object.__new__(SpeculativeDecoder)
    d.gamma = 5
    assert d.estimate_speedup(acceptance_rate=0.8) == pytest.approx(2.7777, abs=0.01)
    d.gamma = 1
    assert d.estimate_speedup(acceptance_rate=0.8) == 1.0


def test_estimate_speedup_edges():
    d = object.__new__(SpeculativeDecoder)
    d.gamma = 5
    assert d.estimate_speedup(acceptance_rate=0.0) == 1.0
    assert d.estimate_speedup(acceptance_rate=1.0) == 5.0


# ---------------------------------------------------------------------------
# Real-model GPU smoke test (opt-in: SPEC_E2E=1)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("SPEC_E2E") != "1" or not torch.cuda.is_available(),
    reason="requires SPEC_E2E=1 on a CUDA host (real models, few minutes)",
)
def test_real_speculative_decoding_smoke():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    draft_path = "Qwen/Qwen2.5-0.5B-Instruct"
    target_path = "Qwen/Qwen3-0.6B"
    target_tok = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
    draft_tok = AutoTokenizer.from_pretrained(draft_path, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    draft = AutoModelForCausalLM.from_pretrained(
        draft_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    target.eval()
    draft.eval()

    decoder = SpeculativeDecoder(
        draft_model=draft,
        draft_tokenizer=draft_tok,
        target_model=target,
        target_tokenizer=target_tok,
        gamma=5,
        device=torch.device("cuda"),
    )
    prompt = target_tok("The capital of France is", return_tensors="pt")[
        "input_ids"
    ].to("cuda")

    ids, stats = decoder.generate(
        prompt, max_tokens=32, eos_token_id=target_tok.eos_token_id
    )

    assert len(ids) > 0
    assert stats.n_target_forwards > 0
    # draft forwards = prefill(1) + proposed (gamma per round) + one re-forward
    # per rejected correction token — so it must be >= proposed + 1
    assert stats.n_draft_proposed > 0
    assert stats.n_draft_forwards >= stats.n_draft_proposed + 1
    text = target_tok.decode(ids, skip_special_tokens=True)
    assert text.strip()
    print(
        f"\n[E2E] acceptance_rate={stats.acceptance_rate:.3f} "
        f"target_fwd={stats.n_target_forwards} draft_fwd={stats.n_draft_forwards} "
        f"text={text[:60]!r}"
    )
