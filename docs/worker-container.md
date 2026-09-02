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
  --label org.opencontainers.image.revision=b95da863d262242e33b8388889baca83a70508a1 \
  --tag freechat-worker:b95da863d262 \
  --load \
  .
```

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
