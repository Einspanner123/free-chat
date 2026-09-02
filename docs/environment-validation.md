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

### Current serving environment

- Editable vLLM fork source: `/home/linkst/workspace/freechat-vllm-fork`, with
  deployment commit `17539077af862832a2804429e9c0fd8896e274bc`, whose
  Anthropic lifecycle changes match authoritative fork commit
  `8681c8040632e0790c1880c90cddb92420189b07`.
- Runtime: `/home/linkst/.venvs/freechat-current`, PyTorch 2.13.0+cu130,
  CUDA runtime 13.0, Triton 3.7.1 and Transformers 5.14.1.
- A4000 serving startup, Chat Completions lifecycle propagation and
  EngineCore allocate/free/hit event emission were exercised with Qwen2.5
  tokenizer/config and vLLM dummy weights. This validates the serving
  mechanism only; it is not a real-model performance run.
- The pinned Qwen2.5-0.5B weight is available at
  `/data/freechat/models/Qwen2.5-0.5B-Instruct/model.safetensors`. Its exact
  size is 988097824 bytes and its SHA-256 is
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
  matching the ross artifact and source blob. Offline validation loaded the
  Qwen2 configuration, 151665-token tokenizer and 290 safetensor keys. A
  separate incomplete tar-stream artifact remains explicitly quarantined and
  is not accepted as a model weight.

On 2026-09-02 the real Qwen2.5-0.5B weight served on the idle A4000 and
returned exact expected text through Chat Completions (`READY`), Responses
(`RESPONSE_OK`) and Anthropic Messages (`MESSAGE_OK`). A second Messages
request returned `LIFECYCLE_OK`; its EngineCore audit contained nine
allocate/free events and the final free event covered block IDs 1, 2 and 3.
Every event preserved tenant, task, session, agent, branch, call, Tool Wait,
worker/cache generation, 500 ms expected resume, priority and offload fields.
The JSONL is archived at
`/media/ross/8TB/linkst/freechat/evidence/20260902/real-model/real-qwen-messages.jsonl`
with SHA-256
`5562bd487d3f870fff199efdfc61d5fc0a3db1b3d0e89b257d4ea6ce2b4908b2`.
This is correctness and integration evidence only, not a latency, throughput
or cache-hit improvement claim.

The host's `/usr/bin/nvcc` is CUDA 11.5 and cannot compile the current
FlashInfer sampling JIT. The successful run therefore set
`VLLM_USE_FLASHINFER_SAMPLER=0` and used the safe non-FlashInfer sampler while
retaining the PyTorch CUDA 13 runtime. CUDA 13 compiler/profiler evidence must
come from the locked worker/profiler image; this host run cannot satisfy that
gate.

### Container remediation status

On 2026-09-02 workstation pulled the upstream baseline
`vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`.
CPU inspection confirmed CUDA compiler 13.0, PyTorch 2.11.0+cu130, Triton
3.6.0, Transformers 5.14.1, vLLM 0.26.0 and FlashInfer 0.6.14. It is therefore
eligible as an upstream baseline but does not match the candidate worker lock.

GPU startup failed before container creation because workstation does not have
`nvidia-container-runtime` or NVIDIA Container Toolkit installed. Both Docker
`--gpus device=1` and the declared `nvidia` runtime are unusable in this state.
The CUDA compiler mismatch is container-solvable, but remains operationally
blocked until the host runtime is installed and Docker is safely restarted.
NVIDIA Container Toolkit 1.20.0 was subsequently installed and Docker was
configured with the NVIDIA runtime. A GPU probe against the digest-pinned
upstream image exposed the A4000, reported CUDA compiler/runtime 13.0 and
successfully executed and synchronized a FlashInfer top-k/top-p sampling
kernel. The image still failed the candidate gate, as intended, because its
PyTorch 2.11, Triton 3.6 and upstream revision do not match the locked fork.

Docker data is mounted at `/data/freechat/docker` on an ext4 volume. The
`overlay2` driver, existing named volumes, digest-pinned images and the etcd,
NATS and scheduler containers recovered after migration.

The constrained-cache mechanism probe is archived under
`/media/ross/8TB/linkst/freechat/evidence/20260901/mechanism`:

- Native probe JSON: `b144dc885c9faf0f5255282b2afb0ceedff6fe8fd99829c5fe37799b76a37fe9`.
- Native cache events: `000df3f08bfc8aefe6ee8152c0cc8c991de2ff3979cd4a6835cf1195c395f0ea`.
- Lifecycle probe JSON: `5195d4e1ad26fb112b28730273a2786546dfd4634a83d2286ffbc61c8ec2077c`.
- Lifecycle cache events: `cb236f27048029ee2125974267178b12ad068e4a4e6a13da06a447b009625607`.

The probe used a 32 MiB KV cache, one 930-token target prompt and four
approximately 1030-token pressure prompts. Native eviction left a 16-token
resume hit; lifecycle-aware ordering retained 928 tokens. The artifact itself
sets `performance_claim_admissible=false`; these values must not populate the
resume placeholders.

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
