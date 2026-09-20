"""Single version-lock reader for source and worker-image validation."""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SHA = r"^[0-9a-f]{40}$"


class GPUStack(BaseModel):
    model_config = ConfigDict(strict=True)
    pytorch: str
    cuda_runtime: str
    triton: str
    transformers: str


class SourcePin(BaseModel):
    model_config = ConfigDict(strict=True)
    upstream_commit: str = Field(pattern=SHA)
    fork_commit: str = Field(pattern=SHA)
    fork_tree: str = Field(pattern=SHA)
    fork_repository: str = Field(pattern=r"^(ssh|https)://[^\s]+$")
    checkout_path: str = Field(pattern=r"^third_party/vllm$")
    worker_image_digest: str


class VersionLock(BaseModel):
    model_config = ConfigDict(strict=True)
    schema_version: int = Field(alias="schema", ge=1, le=1)
    python: str = Field(pattern=r"^3\.12$")
    gpu_stack: GPUStack
    vllm: SourcePin


def load_versions(root: Path = ROOT) -> VersionLock:
    try:
        document = yaml.safe_load((root / "versions.lock.yaml").read_text())
    except yaml.YAMLError as error:
        raise ValueError("invalid_version_lock_yaml") from error
    return VersionLock.model_validate(document)


def expected_worker_versions(root: Path = ROOT) -> dict[str, str]:
    lock = load_versions(root)
    return {
        "python": lock.python,
        "torch": lock.gpu_stack.pytorch,
        "torch_cuda": lock.gpu_stack.cuda_runtime,
        "triton": lock.gpu_stack.triton,
        "transformers": lock.gpu_stack.transformers,
        "fork_revision": lock.vllm.fork_commit,
        "upstream_revision": lock.vllm.upstream_commit,
    }
