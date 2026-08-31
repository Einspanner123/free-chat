# Kernel candidate ledger

No entry in this file is an end-to-end performance claim. Raw artifacts live
outside the Git working tree and are addressed by SHA-256.

## Branch resume block-table gather — rejected

- Environment: NVIDIA RTX A6000, PyTorch 2.11.0+cu130, Triton 3.6.0.
- Shape: source `(4096, 512)`, 32 active rows, int32, 1,000 repetitions.
- PyTorch reference median: 33.992 us.
- Triton candidate median: 39.9335 us.
- Median paired speedup: 0.8516x; the candidate is slower.
- Raw artifact: `/media/ross/8TB/linkst/freechat/evidence/20260831/block-table-a6000-torch211-triton36.json`.
- SHA-256: `2f4a437b42912143b74831d1c65e5e33b34e2982e969408e37dd525bfde9713d`.
- Runtime state: disabled by default; retained only as a falsified candidate.

## Fused per-group KV int8 quantization — microbenchmark passed

- Environment: NVIDIA RTX A6000, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0.
- Shape: 8,388,608 FP16 elements, group size 128, 500 repetitions.
- PyTorch reference median: 605.1840 us.
- Triton candidate median: 97.2800 us.
- Median speedup: 6.2211x.
- Median latency reduction: 507.9040 us; bootstrap 95% CI `[507.9040, 509.4240]` us.
- Raw artifact: `/media/ross/8TB/linkst/freechat/evidence/20260831/kv-quantize-a6000-torch211-triton36-ci.json`.
- SHA-256: `2253071d9d17c0b231aaa6156c0b6444cb9fc085a924f05ab33708c1bd87b1b8`.
- Profiler trace SHA-256: `6a10adffae5720974615db540939ee8e9579bb63c3a901885c7718a98af4f104`.
- Mechanism: profiler shows the reference path launching separate abs, reduction,
  division, round, clamp and cast work; the candidate emits one `_quantize_kernel`.
- Guardrail: disabled by default, group sizes below 128 use the PyTorch reference,
  and unsupported devices, layouts or dtypes fall back safely.
- Replication on workstation with the same locked PyTorch/CUDA/Triton stack:
  - RTX A5000: reference 619.520 us, candidate 95.232 us, 6.5054x;
    latency-reduction bootstrap 95% CI `[524.288, 525.152]` us. Artifact
    `/data/freechat/profiles/20260831/kv-quantize-a5000-torch211-triton36-ci.json`,
    SHA-256 `c44f7eb393e0b7f3607c63f170f2e143d49f7ffdb8ed356dd0ea01b77dbbe735`.
  - RTX A4000: reference 1049.600 us, candidate 136.192 us, 7.7068x;
    latency-reduction bootstrap 95% CI `[913.408, 913.408]` us. Artifact
    `/data/freechat/profiles/20260831/kv-quantize-a4000-torch211-triton36-ci.json`,
    SHA-256 `a506160491b2171c03465d1f2c9c10a41c0a35c9a8b7fd579bd662d9f613d746`.
- Remaining gates: target Harness trace selection, worker-image integration of
  the pinned vLLM hook, end-to-end Agent task confidence interval, non-target
  workload regression, and cancellation/restart behavior. The workstation replication closes only
  the isolated-kernel replication gate; it does not establish an Agent-task or
  serving-system improvement.
