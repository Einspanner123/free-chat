# Worker container build and acceptance

The worker container is the only accepted CUDA compiler and runtime boundary.
Host `nvcc`, Python packages and profiler binaries are not inherited as evidence.

## Build from the pinned fork

Build from the root of the vLLM fork at the commit in `versions.lock.yaml`:

```bash
docker buildx build \
  --file docker/Dockerfile \
  --target vllm-openai \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg torch_cuda_arch_list=8.6 \
  --build-arg VLLM_USE_PRECOMPILED=1 \
  --build-arg VLLM_MERGE_BASE_COMMIT=9c22668436a4d94aab87ea74a220e060415cf1d8 \
  --build-arg VLLM_MAIN_CUDA_VERSION=13.0 \
  --label org.opencontainers.image.revision=8e78a3c613072632aa822c9aed2f698e76046219 \
  --label io.freechat.vllm.upstream-revision=9c22668436a4d94aab87ea74a220e060415cf1d8 \
  --tag freechat-worker:8e78a3c61307 \
  --load \
  .
```

The fork changes Python lifecycle, scheduling and cache-policy integration but
does not modify vLLM C++ or CUDA sources. The build therefore reuses the
commit-specific CUDA 13 extension wheel published for the exact upstream
commit recorded in `versions.lock.yaml`, while packaging the fork's Python and
Rust layers from source. Both revisions are OCI labels and acceptance checks;
using a wheel from a different upstream commit fails closed. A source build
remains available for auditing, but is not required merely to rebuild unchanged
upstream kernels.

The build is not release evidence. Push it to the project registry, resolve the
repository digest, and replace `worker_image_digest: UNRESOLVED` only after the
GPU acceptance command succeeds against the digest reference.

## Fail-closed acceptance

```bash
uv run python -m tools.validate_worker_image \
  registry.example/freechat-worker@sha256:REPLACE_ME \
  --gpu 1 \
  --output /data/freechat/profiles/worker-image.json
```

Acceptance requires all of the following in one run:

- Python 3.12, PyTorch 2.13.0+cu130, CUDA runtime and `nvcc` 13.0,
  Triton 3.7.1 and Transformers 5.16.1.
- A CUDA-visible physical GPU and a real FlashInfer top-k/top-p sampling kernel
  invocation, including synchronization so JIT/compiler failures are observable.
- The authoritative fork revision in the OCI image label.
- The exact upstream revision that supplied the precompiled CUDA extensions.
- An immutable repository digest rather than a mutable local tag.

The upstream `vllm/vllm-openai:v0.26.0` image is retained only as a direct-vLLM
baseline. Its PyTorch 2.11 and Triton 3.6 packages do not satisfy the candidate
worker lock.

## Host prerequisite

Docker must expose the NVIDIA runtime before the GPU gate can run. Installing or
reconfiguring NVIDIA Container Toolkit and restarting Docker are host operations;
they must be scheduled so unrelated running containers are not interrupted. A
CPU-only container probe or a successful image pull does not satisfy this gate.
Omitting `--gpu` always produces a non-accepted inspection report.

The local candidate passed every GPU and locked-runtime check on an A4000 on
2026-09-07, including synchronized FlashInfer sampling. It remains unaccepted
because no project registry is configured and the local tag has no repository
digest. A registry push and a repeat of the same command against the resolved
digest are still required; the local image ID is not a substitute.
