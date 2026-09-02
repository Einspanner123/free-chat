from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED = {
    "python": "3.12",
    "torch": "2.13.0+cu130",
    "torch_cuda": "13.0",
    "triton": "3.7.1",
    "transformers": "5.16.1",
    "fork_revision": "8e78a3c613072632aa822c9aed2f698e76046219",
}


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


Runner = Callable[[Sequence[str]], CommandResult]


def run_command(command: Sequence[str]) -> CommandResult:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return CommandResult(completed.stdout, completed.stderr, completed.returncode)


def _decode_json(result: CommandResult, action: str) -> Any:
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{action} failed: {message}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{action} returned invalid JSON") from error


def inspect_image(image: str, runner: Runner) -> dict[str, Any]:
    payload = _decode_json(runner(["docker", "image", "inspect", image]), "image inspect")
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError("image inspect did not return exactly one image")
    inspected = payload[0]
    if not isinstance(inspected, dict):
        raise RuntimeError("image inspect returned an invalid record")
    return inspected


def probe_image(image: str, gpu: str | None, runner: Runner) -> dict[str, Any]:
    script = """
import json, platform, re, subprocess
import flashinfer, torch, transformers, triton, vllm
nvcc = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True).stdout
match = re.search(r"release ([0-9]+[.][0-9]+)", nvcc)
device = None
flashinfer_sampling = False
if torch.cuda.is_available():
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    probabilities = torch.softmax(torch.randn(2, 128, device="cuda"), dim=-1)
    samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probabilities, 32, 0.9, deterministic=True
    )
    torch.cuda.synchronize()
    flashinfer_sampling = tuple(samples.shape) == (2,)
    device = {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "flashinfer": flashinfer.__version__,
    "nvcc_cuda": match.group(1) if match else None,
    "cuda_available": torch.cuda.is_available(),
    "flashinfer_sampling": flashinfer_sampling,
    "device": device,
}))
""".strip()
    command = ["docker", "run", "--rm"]
    if gpu is not None:
        command.extend(["--gpus", f"device={gpu}"])
    command.extend(["--entrypoint", "python3", image, "-c", script])
    payload = _decode_json(runner(command), "worker image probe")
    if not isinstance(payload, dict):
        raise RuntimeError("worker image probe returned an invalid record")
    return payload


def build_report(
    *,
    image: str,
    inspected: dict[str, Any],
    probe: dict[str, Any],
    gpu_required: bool,
) -> dict[str, Any]:
    config = inspected.get("Config") or {}
    labels = config.get("Labels") or {}
    repo_digests = inspected.get("RepoDigests") or []
    actual = {
        "python": str(probe.get("python", ""))[:4],
        "torch": probe.get("torch"),
        "torch_cuda": probe.get("torch_cuda"),
        "triton": probe.get("triton"),
        "transformers": probe.get("transformers"),
        "fork_revision": labels.get("org.opencontainers.image.revision"),
    }
    checks = {
        name: {
            "expected": expected,
            "actual": actual.get(name),
            "passed": actual.get(name) == expected,
        }
        for name, expected in EXPECTED.items()
    }
    checks["nvcc_cuda"] = {
        "expected": EXPECTED["torch_cuda"],
        "actual": probe.get("nvcc_cuda"),
        "passed": probe.get("nvcc_cuda") == EXPECTED["torch_cuda"],
    }
    checks["gpu_runtime"] = {
        "expected": True,
        "actual": bool(probe.get("cuda_available")),
        "passed": bool(probe.get("cuda_available")),
    }
    checks["flashinfer_sampling"] = {
        "expected": True,
        "actual": bool(probe.get("flashinfer_sampling")),
        "passed": bool(probe.get("flashinfer_sampling")),
    }
    checks["immutable_repo_digest"] = {
        "expected": True,
        "actual": bool(repo_digests),
        "passed": bool(repo_digests),
    }
    return {
        "schema": 1,
        "observed_at": datetime.now(UTC).isoformat(),
        "image_reference": image,
        "image_id": inspected.get("Id"),
        "repo_digests": repo_digests,
        "gpu_requested": gpu_required,
        "probe": probe,
        "checks": checks,
        "accepted": all(check["passed"] for check in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed FreeChat worker image gate")
    parser.add_argument("image", help="immutable image reference to validate")
    parser.add_argument("--gpu", help="Docker GPU device index; omit only for CPU inspection")
    parser.add_argument("--output", type=Path, help="optional JSON evidence path")
    arguments = parser.parse_args()

    try:
        inspected = inspect_image(arguments.image, run_command)
        probe = probe_image(arguments.image, arguments.gpu, run_command)
        report = build_report(
            image=arguments.image,
            inspected=inspected,
            probe=probe,
            gpu_required=arguments.gpu is not None,
        )
    except RuntimeError as error:
        print(json.dumps({"accepted": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n")
    print(encoded)
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
