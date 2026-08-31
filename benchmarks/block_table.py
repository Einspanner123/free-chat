from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime

import torch
from freechat_worker.kernels import gather_block_rows, gather_block_rows_reference


def measure(operation: object, warmup: int, repetitions: int) -> list[float]:
    callable_operation = operation
    for _ in range(warmup):
        callable_operation()  # type: ignore[operator]
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        callable_operation()  # type: ignore[operator]
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    table = torch.randint(
        0,
        1_000_000,
        (arguments.rows, arguments.width),
        device="cuda",
        dtype=torch.int32,
    )
    indices = torch.randint(
        0,
        arguments.rows,
        (arguments.active,),
        device="cuda",
        dtype=torch.int32,
    )
    reference = measure(
        lambda: gather_block_rows_reference(table, indices), arguments.warmup, arguments.repetitions
    )
    candidate = measure(
        lambda: gather_block_rows(table, indices), arguments.warmup, arguments.repetitions
    )
    speedups = [
        baseline / optimized
        for baseline, optimized in zip(reference, candidate, strict=True)
    ]
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": arguments.rows,
        "width": arguments.width,
        "active": arguments.active,
        "repetitions": arguments.repetitions,
        "reference_median_us": statistics.median(reference),
        "candidate_median_us": statistics.median(candidate),
        "median_paired_speedup": statistics.median(speedups),
        "raw_reference_us": reference,
        "raw_candidate_us": candidate,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
