# PyTorch and vLLM stack conflict

- Status: `RESOLVED`
- Detected: 2026-09-01 Asia/Shanghai
- Resolved: 2026-09-01 Asia/Shanghai
- Owner decision: Option A; align every worker and benchmark to the pinned
  vLLM PyTorch/CUDA/Triton stack and rebuild the evidence baseline.
- Scope blocked: worker image, three-model baseline, Harness end-to-end metrics,
  kernel-to-serving comparison, resume/cancellation fault tests.

## Conflicting constraints

The FreeChat implementation contract and `versions.lock.yaml` pin PyTorch
2.11.0+cu130 and Triton 3.6.0. The pinned vLLM upstream/fork commit requires
PyTorch 2.13.0 and its verified development environment runs
PyTorch 2.13.0+cu130 with Triton 3.7.1. Both use CUDA 13.0 and Transformers
5.14.1, but the PyTorch/Triton ABI and generated kernels are not the same
experimental stack.

Evidence:

- vLLM `requirements/cuda.txt`: `torch==2.13.0`.
- vLLM `requirements/build/cuda.txt`: `torch==2.13.0`.
- vLLM `requirements/test/cuda.txt`: `torch==2.13.0+cu130`.
- vLLM fork environment: PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1,
  Transformers 5.14.1.
- FreeChat locked GPU environment and kernel evidence: PyTorch 2.11.0+cu130,
  CUDA 13.0, Triton 3.6.0, Transformers 5.14.1.

## Options

### A — align FreeChat to the pinned vLLM stack (recommended)

Lock PyTorch 2.13.0+cu130 and Triton 3.7.1 for all worker images and rerun every
GPU correctness test, kernel benchmark and profiler trace. Existing 6.2–7.7x
kernel results remain historical evidence until replicated on the new lock;
they cannot be mixed with new end-to-end metrics.

This keeps the current vLLM fork base and minimizes long-term divergence from
the inference engine, at the cost of invalidating the current GPU performance
baseline until it is rerun.

### B — select and maintain a vLLM commit compatible with PyTorch 2.11

Find a compatible upstream commit, replay the narrow lifecycle/cache changes,
and rerun the fork test suite. This preserves the current kernel stack but
creates a larger maintenance delta and may lose engine features already used
by the current hook integration.

### C — maintain separate kernel and serving stacks

Keep PyTorch 2.11/Triton 3.6 for kernel evidence and use PyTorch 2.13/Triton 3.7
for vLLM serving. This is not recommended for resume claims because isolated
kernel and end-to-end results would not share one deployable artifact.

## Exit condition

Only an explicit owner selection closes this conflict. After selection, record
the answer here, set status to `RESOLVED`, update the lock and rebuild the
evidence baseline before producing resume metrics.

The owner selected option A. The dependency lock may advance immediately, but
performance claims remain blocked until all three GPUs have reproduced the
correctness, profiler and benchmark artifacts under the resolved stack.
