# Environment validation

This document separates host observations from evidence produced by the locked
worker stack. It does not treat an installed executable or a configuration file
as proof that a workload used that toolchain.

## ross

- GPU: one NVIDIA RTX A6000, 48 GB, compute capability 8.6.
- Locked GPU test environment: PyTorch 2.11.0+cu130, CUDA runtime 13.0,
  Triton 3.6.0.
- Host topology artifact:
  `/media/ross/8TB/linkst/freechat/evidence/20260831/topology-ross.json`.
- Topology SHA-256:
  `229ab9161ad5a51354ff09a0bfa7010a551c0a8133c4ae82f2a6f2db05ef710e`.
- The host does not expose `nvcc`, `ncu`, or `nsys`; profiler claims must name
  the locked image or environment that produced them.

## workstation

- GPUs: NVIDIA RTX A5000 24 GB and NVIDIA RTX A4000 16 GB; both compute
  capability 8.6.
- Host `nvcc`: CUDA 11.5 (`V11.5.119`). Host Nsight Compute: 2021.3.1.
  Nsight Systems was not found. These host tools are explicitly excluded as
  CUDA 13 or current-profiler evidence.
- The measured P2P matrix reports `NS` in both directions between the A5000 and
  A4000. Tensor/pipeline parallel execution on these two GPUs therefore remains
  rejected until a separate communication and end-to-end experiment proves a
  benefit; the current observation is not such proof.
- Locked GPU validation environment: `/home/linkst/.venvs/freechat-079a65e`.
  It was used because `/data` permits data writes but rejects the metadata
  operations required to build a Python environment. Models, profiles and raw
  evidence remain under `/data/freechat`.
- Topology artifact:
  `/data/freechat/profiles/20260831/topology-workstation.json`.
- Topology SHA-256:
  `61c2ecd59574c7c7417b838c83dd2fa86b50067d644b972cade6386f429b7624`.

## Control-plane recovery integration

On workstation, the digest-pinned Compose services started a real etcd 3.6.5,
NATS JetStream 2.14.6 and the scheduler control image. The validation sequence
registered a worker, created a route lease, restarted the scheduler process,
restored the worker and decision from etcd, explained the original decision,
and released its lease. The second release event reused its deterministic
message ID; JetStream retained five messages rather than accepting a duplicate.

- Scheduler image ID:
  `sha256:4357f020d26d0ad2a091eff79b6fb9c211147b305cd324e4419b9d25dbbab443`.
- After release, etcd retained the topology-generation and worker records; no
  lease or outbox key remained.
- JetStream stream `FREECHAT_LIFECYCLE` retained five messages across three
  subjects. This validates one scheduler replica and persistent dependencies,
  not production HA or network-partition behavior.
