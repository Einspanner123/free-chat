from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch  # type: ignore[import-not-found]
from freechat_worker.kernels import quantize_kv, quantize_kv_reference
from kv_quantize import query_gpu_state, require_exclusive_gpu  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    arguments = parser.parse_args()
    require_exclusive_gpu()
    before = query_gpu_state()
    source = torch.randn(arguments.elements, device="cuda", dtype=torch.float16)
    for _ in range(arguments.warmup):
        quantize_kv_reference(source, arguments.group_size)
        quantize_kv(source, arguments.group_size, enable_triton=True)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        with torch.profiler.record_function("freechat::kv_quantize_reference"):
            quantize_kv_reference(source, arguments.group_size)
        torch.cuda.synchronize()
        with torch.profiler.record_function("freechat::kv_quantize_triton"):
            quantize_kv(source, arguments.group_size, enable_triton=True)
        torch.cuda.synchronize()
    arguments.trace.parent.mkdir(parents=True, exist_ok=True)
    arguments.summary.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(arguments.trace))
    events = [
        {
            "name": event.key,
            "count": event.count,
            "self_device_time_us": float(event.self_device_time_total),
            "device_time_us": float(event.device_time_total),
        }
        for event in profiler.key_averages()
    ]
    arguments.summary.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_state_before": before,
                "gpu_state_after": query_gpu_state(),
                "elements": arguments.elements,
                "group_size": arguments.group_size,
                "events": events,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
