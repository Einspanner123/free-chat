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
- Remaining gates: target Harness trace selection, vLLM hook integration,
  end-to-end Agent task confidence interval, non-target workload regression,
  cancellation/restart behavior, and workstation GPU replication.
