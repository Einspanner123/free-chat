from __future__ import annotations

from typing import Any

from tools.validate_worker_image import EXPECTED, build_report


def valid_probe() -> dict[str, Any]:
    return {
        "python": "3.12.11",
        "torch": EXPECTED["torch"],
        "torch_cuda": EXPECTED["torch_cuda"],
        "triton": EXPECTED["triton"],
        "transformers": EXPECTED["transformers"],
        "nvcc_cuda": EXPECTED["torch_cuda"],
        "cuda_available": True,
        "flashinfer_sampling": True,
        "device": {"name": "NVIDIA RTX A4000"},
    }


def valid_inspection() -> dict[str, Any]:
    return {
        "Id": "sha256:candidate",
        "RepoDigests": ["registry.example/freechat@sha256:candidate"],
        "Config": {
            "Labels": {
                "org.opencontainers.image.revision": EXPECTED["fork_revision"],
                "io.freechat.vllm.upstream-revision": EXPECTED["upstream_revision"],
            }
        },
    }


def test_accepts_only_complete_locked_gpu_image() -> None:
    report = build_report(
        image="registry.example/freechat@sha256:candidate",
        inspected=valid_inspection(),
        probe=valid_probe(),
        gpu_required=True,
    )

    assert report["accepted"] is True
    assert all(check["passed"] for check in report["checks"].values())


def test_rejects_upstream_stack_even_when_cuda_is_available() -> None:
    probe = valid_probe()
    probe["torch"] = "2.11.0+cu130"
    probe["triton"] = "3.6.0"

    report = build_report(
        image="vllm/vllm-openai:v0.26.0",
        inspected=valid_inspection(),
        probe=probe,
        gpu_required=True,
    )

    assert report["accepted"] is False
    assert report["checks"]["torch"]["passed"] is False
    assert report["checks"]["triton"]["passed"] is False


def test_rejects_local_tag_without_immutable_digest() -> None:
    inspected = valid_inspection()
    inspected["RepoDigests"] = []

    report = build_report(
        image="freechat-worker:local",
        inspected=inspected,
        probe=valid_probe(),
        gpu_required=True,
    )

    assert report["accepted"] is False
    assert report["checks"]["immutable_repo_digest"]["passed"] is False


def test_rejects_image_without_exact_upstream_revision() -> None:
    inspected = valid_inspection()
    inspected["Config"]["Labels"]["io.freechat.vllm.upstream-revision"] = "wrong"

    report = build_report(
        image="registry.example/freechat@sha256:candidate",
        inspected=inspected,
        probe=valid_probe(),
        gpu_required=True,
    )

    assert report["accepted"] is False
    assert report["checks"]["upstream_revision"]["passed"] is False


def test_cpu_inspection_does_not_masquerade_as_gpu_acceptance() -> None:
    probe = valid_probe()
    probe["cuda_available"] = False
    probe["device"] = None

    report = build_report(
        image="registry.example/freechat@sha256:candidate",
        inspected=valid_inspection(),
        probe=probe,
        gpu_required=True,
    )

    assert report["accepted"] is False
    assert report["checks"]["gpu_runtime"]["passed"] is False


def test_omitting_gpu_never_produces_an_accepted_report() -> None:
    probe = valid_probe()
    probe["cuda_available"] = False
    probe["flashinfer_sampling"] = False
    probe["device"] = None

    report = build_report(
        image="registry.example/freechat@sha256:candidate",
        inspected=valid_inspection(),
        probe=probe,
        gpu_required=False,
    )

    assert report["accepted"] is False
    assert report["gpu_requested"] is False
    assert report["checks"]["gpu_requested"]["passed"] is False
    assert report["checks"]["flashinfer_sampling"]["passed"] is False
