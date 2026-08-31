# FreeChat pre-rewrite environment archive

Captured: 2026-08-31 Asia/Shanghai

## Repository

- Remote: `https://github.com/Einspanner123/free-chat.git`
- Branch: `main`
- Source commit before plan commit: `ee5a67d`
- Source summary: `docs: add audit-ready BENCHMARKS.md mapping every headline number to script + result file`
- Worktree additions before archive: `.hermes/plans/agent-aware-inference-infra-plan.md`, `.hermes/plans/production-inference-platform-plan.md`
- Recovery policy: keep the immutable archive tag and archive branch; do not treat historical benchmark claims as target-system evidence.

## ross

- OS: Ubuntu 24.04
- GPU: NVIDIA RTX A6000, 49,140 MiB, compute capability 8.6
- Driver: 580.173.02
- Python: 3.12.3
- Project environment: PyTorch 2.11.0+cu130, CUDA runtime 13.0, Transformers 5.14.1, vLLM 0.26.0, Triton 3.6.0, pytest 9.1.1
- NCCL reported by PyTorch: 2.28.9
- Docker: 29.1.3
- Missing at capture: Docker Compose plugin, NVIDIA container runtime, `nvcc`, Nsight Systems, Nsight Compute, Go toolchain
- Root filesystem: 117 GiB total, 17 GiB available at capture
- Data filesystem: `/media/ross/8TB`, 7.3 TiB total, 1.7 TiB available at capture

## workstation

- OS: Ubuntu 22.04
- GPUs: NVIDIA RTX A5000 24,564 MiB and NVIDIA RTX A4000 16,376 MiB, compute capability 8.6
- Driver: 580.173.02
- GPU topology: PHB, no NVLink
- Docker Compose: 5.1.3
- NVIDIA container runtime: installed
- Host CUDA compiler: 11.5; host Nsight Compute: 2021.3.1; neither is valid evidence for the CUDA 13 worker environment
- Root filesystem: 916 GiB total, 36 GiB available at capture
- Data filesystem: `/data`, 3.7 TiB total, 3.0 TiB available at capture
- Project Python environment was not installed at capture.

## Network and execution boundary

- Both hosts expose physical network and Tailscale interfaces.
- The observed remote path may use Tailscale DERP and must not be assumed to support cross-node tensor, pipeline, expert-parallel collectives, or hot KV migration.
- The target system must measure RTT, jitter, loss and effective bandwidth before selecting a network-dependent plan.
- The three GPUs are independent workers by default.

## Test snapshot

- `services/llm-inference`: 197 passed, 4 skipped.
- Skipped real-GPU gates: KV compression, KV eviction, prefix reuse and speculative decoding.
- `inference-engine/tests`: 39 passed; these tests validate an in-memory Python block-ID model rather than real vLLM GPU KV ownership.
- Go tests could not run because the target host had no Go toolchain.

## Archive contents required before source removal

- All tracked source at the archive commit.
- Both implementation plans.
- `research/**/results/*` ignored result files.
- `docs/BENCHMARKS.md`, README files and resolved dependency manifests.
- This environment archive.
- SHA-256 inventory for the filesystem evidence bundle.
