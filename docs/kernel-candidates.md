# Kernel candidate ledger

No entry in this file is an end-to-end performance claim. Raw artifacts live
outside the Git working tree and are addressed by SHA-256.

The 2026-08-31 measurements below used PyTorch 2.11/Triton 3.6. After the
2026-09-01 owner decision aligned the deployable worker with the pinned vLLM
stack, these measurements became historical evidence. They cannot be quoted as
current project performance until replicated under PyTorch 2.13/Triton 3.7.

## Current stack replication status

- The A6000 correctness matrix passed for FP16, BF16 and FP32 at group sizes
  32, 128 and 256 under PyTorch 2.13.0+cu130 and Triton 3.7.1.
- The first A6000 timing run is `INVALID/CONTENDED`: another training process
  occupied the GPU, which was at P2, about 196 W and 84 C during inspection.
  Its artifact is retained as
  `/media/ross/8TB/linkst/freechat/evidence/20260901/INVALID-contended-kv-quantize-a6000-torch213-triton371.json`
  with SHA-256
  `1ca81026e5a9c60ddba39f3a7c1da36524c7ee19cf218f69a33ac438036cf9ae`.
  The observed value is forbidden in claims.
- The benchmark now refuses to run when another compute process occupies the
  selected GPU and records pre/post temperature, power, clocks and P-state.
- Valid timing, profiler and Agent-task evidence remain open.

### A4000 current-stack evidence

- Environment: RTX A4000, driver 580.173.02, PyTorch 2.13.0+cu130,
  CUDA 13.0 and Triton 3.7.1. The GPU was exclusive for every accepted run.
- Correctness: all 9 combinations of FP16/BF16/FP32 and group sizes
  32/128/256 passed against the PyTorch reference.
- Shape: 8,388,608 FP16 elements, group size 128, 50 warm-ups and 500
  randomized paired measurements per trial.
- Three trial medians: reference 1037.31--1038.34 us, candidate
  119.81--131.07 us, or 7.92--8.66x isolated speedup. Paired-bootstrap
  latency-reduction intervals were positive in all three trials.
- Raw artifact SHA-256 values:
  - seed 20260901: `c0ee3c76cc683bc828fc40bead3e107abe1f9d5fef621ff816fb8b4f89a83ea9`;
  - seed 20260902: `35ebbaca677ea55a98cc18c2a115c99a8876e49d824acc5478772cbd4ec106f7`;
  - seed 20260903: `a9e779bb94d87a68f12d71be63c2abb3d09021d05f757a88b34de62409f4eb3b`.
- Current-stack profiler: the reference launched separate `abs`, `amax`,
  `div`, `round` and `clamp` operations; the candidate launched one
  `_quantize_kernel`. Summary SHA-256
  `76855739b122a34751402b58a90674b1ecced1940991e25437bac58c4a4ffd16`;
  Chrome trace SHA-256
  `ded731702c7d7148edbe2a78da5744ffa38b5108557747af0b5ee9064fee4b2d`.
- Environment manifests: pip freeze SHA-256
  `96934ddec944c4e08b6e235c4c491e5c598fc0cc4fe1e58ad9d364f71672cc02`;
  full `nvidia-smi -q` SHA-256
  `e1b76ca9fb14f0b87be5d49da2f6ba2dd4a806e12f4b4de54ddd32b6ba69aa0f`.
- This closes only one-GPU correctness, repeatability and profiler-mechanism
  evidence. A6000/A5000 replication, serving integration, non-target guardrail
  and Agent-task end-to-end gates remain open, so Claims Ledger stays
  `UNVERIFIED`.

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
