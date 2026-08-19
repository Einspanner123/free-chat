"""Real-model E2E for prefix KV reuse on the HF engine (transformers 5.x).

Validates the forward-resume + manual decode loop against a greedy
`model.generate(do_sample=False)` baseline, for both a cache MISS (cold) and a
HIT (second identical request). Gated behind KV_E2E=1 on a CUDA host -- this is
the validation that was missing and let the broken `generate(suffix, past=...)`
resume ship.
"""

import os
import sys

import pytest

# Other test files install a MODULE-LEVEL MOCK torch/transformers into
# sys.modules; evict them so this file gets the REAL modules (mirror
# test_kv_compression.py).
_torch = sys.modules.get("torch")
if _torch is not None and not getattr(_torch, "__file__", None):
    del sys.modules["torch"]
import torch  # noqa: E402

_tf = sys.modules.get("transformers")
if _tf is not None and not getattr(_tf, "__file__", None):
    del sys.modules["transformers"]

sys.path.insert(0, "/home/linkst/workspace/projects/free-chat/services/llm-inference/src")
from engine_base import EngineConfig  # noqa: E402
from hf_engine import HFEngine  # noqa: E402


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("KV_E2E") != "1" or not torch.cuda.is_available(),
    reason="requires KV_E2E=1 on a CUDA host (real model)",
)
def test_prefix_reuse_matches_baseline_greedy():
    cfg = EngineConfig(
        model_path="Qwen/Qwen3-0.6B",
        prefix_cache_enabled=True,
        prefix_cache_capacity=4,
    )
    eng = HFEngine(config=cfg)
    try:
        msgs = [{"role": "user", "content": "The history of the railway is the history of iron and coal. " * 4}]
        text = eng._format_messages(msgs)
        ids = eng.tokenizer(text, return_tensors="pt").input_ids.to(eng.device)
        n = ids.shape[1]

        # Baseline: greedy generation via model.generate(do_sample=False).
        with torch.no_grad():
            base = eng.model.generate(
                ids, do_sample=False, max_new_tokens=8, pad_token_id=eng.tokenizer.eos_token_id
            )
        base_txt = eng.tokenizer.decode(base[0, n:], skip_special_tokens=True)

        # MISS (cold cache): prefix path, pure greedy (rep=1.0) to match the
        # do_sample=False baseline oracle.
        r1 = eng.generate(msgs, temperature=0, repetition_penalty=1.0, max_tokens=8)
        assert r1.chunk == base_txt, (
            f"prefix MISS != baseline: {r1.chunk!r} vs {base_txt!r}"
        )

        # HIT (second identical request): must reproduce the same output.
        assert len(eng._prefix_cache) == 1, "prefix KV was not stored after MISS"
        r2 = eng.generate(msgs, temperature=0, repetition_penalty=1.0, max_tokens=8)
        assert r2.chunk == base_txt, (
            f"prefix HIT != baseline: {r2.chunk!r} vs {base_txt!r}"
        )

        # A different prompt sharing a PREFIX of the first should also succeed
        # (partial hit) and produce its own greedy baseline.
        msgs2 = [{"role": "user", "content": "The history of the railway is the history of iron and coal. " * 4 + " Trains changed everything."}]
        text2 = eng._format_messages(msgs2)
        ids2 = eng.tokenizer(text2, return_tensors="pt").input_ids.to(eng.device)
        n2 = ids2.shape[1]
        with torch.no_grad():
            base2 = eng.model.generate(
                ids2, do_sample=False, max_new_tokens=8, pad_token_id=eng.tokenizer.eos_token_id
            )
        base2_txt = eng.tokenizer.decode(base2[0, n2:], skip_special_tokens=True)
        r3 = eng.generate(msgs2, temperature=0, repetition_penalty=1.0, max_tokens=8)
        assert r3.chunk == base2_txt, (
            f"prefix partial-hit != baseline: {r3.chunk!r} vs {base2_txt!r}"
        )
    finally:
        eng.close()
