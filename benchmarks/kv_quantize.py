from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import UTC, datetime

import torch
from freechat_worker.kernels import quantize_kv, quantize_kv_reference


def measure(operation: object, warmup: int, repetitions: int) -> list[float]:
    callable_operation = operation
    for _ in range(warmup):
        callable_operation()  # type: ignore[operator]
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_operation()  # type: ignore[operator]
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000)
    return samples


def bootstrap_latency_reduction_ci(
    reference: list[float],
    candidate: list[float],
    *,
    samples: int = 5_000,
    seed: int = 20_260_831,
) -> tuple[float, float]:
    generator = random.Random(seed)
    reductions: list[float] = []
    for _ in range(samples):
        reference_median = statistics.median(generator.choices(reference, k=len(reference)))
        candidate_median = statistics.median(generator.choices(candidate, k=len(candidate)))
        reductions.append(reference_median - candidate_median)
    reductions.sort()
    return reductions[int(samples * 0.025)], reductions[int(samples * 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    arguments = parser.parse_args()
    source = torch.randn(arguments.elements, device="cuda", dtype=torch.float16)
    reference = measure(
        lambda: quantize_kv_reference(source, arguments.group_size),
        arguments.warmup,
        arguments.repetitions,
    )
    candidate = measure(
        lambda: quantize_kv(source, arguments.group_size, enable_triton=True),
        arguments.warmup,
        arguments.repetitions,
    )
    reduction_lower, reduction_upper = bootstrap_latency_reduction_ci(reference, candidate)
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "elements": arguments.elements,
        "group_size": arguments.group_size,
        "repetitions": arguments.repetitions,
        "reference_median_us": statistics.median(reference),
        "candidate_median_us": statistics.median(candidate),
        "median_speedup": statistics.median(reference) / statistics.median(candidate),
        "median_latency_reduction_us": statistics.median(reference)
        - statistics.median(candidate),
        "latency_reduction_us_ci95": [reduction_lower, reduction_upper],
        "confidence_method": "independent bootstrap of medians, 5000 resamples, seed 20260831",
        "raw_reference_us": reference,
        "raw_candidate_us": candidate,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
