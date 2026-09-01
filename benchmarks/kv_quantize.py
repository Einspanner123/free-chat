from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime

import torch  # type: ignore[import-not-found]
import triton  # type: ignore[import-not-found]
from freechat_worker.kernels import quantize_kv, quantize_kv_reference


def query_gpu_state() -> dict[str, str]:
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", maxsplit=1)[0]
    query = (
        "uuid,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,"
        "clocks.sm,clocks.mem"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={visible_device}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [item.strip() for item in completed.stdout.strip().split(",")]
    keys = query.split(",")
    if len(keys) != len(values):
        raise RuntimeError("unexpected nvidia-smi GPU state output")
    return dict(zip(keys, values, strict=True))


def require_exclusive_gpu() -> None:
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", maxsplit=1)[0]
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={visible_device}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    contenders = [line for line in completed.stdout.splitlines() if line.strip()]
    if contenders:
        raise RuntimeError(f"exclusive GPU required; active compute processes: {contenders}")


def measure_once(operation: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1_000)


def measure_paired(
    reference: Callable[[], object],
    candidate: Callable[[], object],
    warmup: int,
    repetitions: int,
    seed: int,
) -> tuple[list[float], list[float], list[str]]:
    for _ in range(warmup):
        reference()
        candidate()
    torch.cuda.synchronize()
    generator = random.Random(seed)
    reference_samples: list[float] = []
    candidate_samples: list[float] = []
    pair_order: list[str] = []
    for _ in range(repetitions):
        candidate_first = bool(generator.getrandbits(1))
        if candidate_first:
            candidate_samples.append(measure_once(candidate))
            reference_samples.append(measure_once(reference))
            pair_order.append("candidate-reference")
        else:
            reference_samples.append(measure_once(reference))
            candidate_samples.append(measure_once(candidate))
            pair_order.append("reference-candidate")
    return reference_samples, candidate_samples, pair_order


def bootstrap_latency_reduction_ci(
    reference: list[float],
    candidate: list[float],
    *,
    samples: int = 5_000,
    seed: int = 20_260_831,
) -> tuple[float, float]:
    if len(reference) != len(candidate):
        raise ValueError("paired samples must have equal length")
    generator = random.Random(seed)
    paired_reductions = [left - right for left, right in zip(reference, candidate, strict=True)]
    reductions: list[float] = []
    for _ in range(samples):
        reductions.append(
            statistics.median(
                generator.choices(paired_reductions, k=len(paired_reductions))
            )
        )
    reductions.sort()
    return reductions[int(samples * 0.025)], reductions[int(samples * 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--allow-contended", action="store_true")
    arguments = parser.parse_args()
    if not arguments.allow_contended:
        require_exclusive_gpu()
    gpu_state_before = query_gpu_state()
    source = torch.randn(arguments.elements, device="cuda", dtype=torch.float16)
    reference, candidate, pair_order = measure_paired(
        lambda: quantize_kv_reference(source, arguments.group_size),
        lambda: quantize_kv(source, arguments.group_size, enable_triton=True),
        arguments.warmup,
        arguments.repetitions,
        arguments.seed,
    )
    reduction_lower, reduction_upper = bootstrap_latency_reduction_ci(
        reference,
        candidate,
        seed=arguments.seed,
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu_state_after = query_gpu_state()
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "gpu_state_before": gpu_state_before,
        "gpu_state_after": gpu_state_after,
        "exclusive_gpu_required": not arguments.allow_contended,
        "elements": arguments.elements,
        "group_size": arguments.group_size,
        "repetitions": arguments.repetitions,
        "seed": arguments.seed,
        "reference_median_us": statistics.median(reference),
        "candidate_median_us": statistics.median(candidate),
        "median_speedup": statistics.median(reference) / statistics.median(candidate),
        "median_latency_reduction_us": statistics.median(reference)
        - statistics.median(candidate),
        "latency_reduction_us_ci95": [reduction_lower, reduction_upper],
        "confidence_method": "paired bootstrap of latency reductions, 5000 resamples",
        "pair_order": pair_order,
        "raw_reference_us": reference,
        "raw_candidate_us": candidate,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
