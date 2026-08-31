from __future__ import annotations

import os
from typing import Any


def quantize_kv_reference(source: Any, group_size: int = 128) -> tuple[Any, Any]:
    import torch  # type: ignore[import-not-found]

    _validate(source, group_size)
    groups = source.reshape(-1, group_size).float()
    maximum = groups.abs().amax(dim=1)
    scales = torch.clamp(maximum / 127.0, min=1e-12)
    quantized = torch.round(groups / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.reshape(source.shape), scales


def quantize_kv(
    source: Any,
    group_size: int = 128,
    *,
    enable_triton: bool | None = None,
) -> tuple[Any, Any]:
    import torch

    _validate(source, group_size)
    enabled = (
        os.environ.get("FREECHAT_ENABLE_TRITON_KV_QUANTIZE") == "1"
        if enable_triton is None
        else enable_triton
    )
    if (
        not enabled
        or group_size < 128
        or not source.is_cuda
        or not source.is_contiguous()
        or source.dtype not in {torch.float16, torch.bfloat16, torch.float32}
    ):
        return quantize_kv_reference(source, group_size)
    from freechat_worker.kernels.triton_kv_quantize import launch_quantize

    quantized = torch.empty_like(source, dtype=torch.int8)
    scales = torch.empty(source.numel() // group_size, device=source.device, dtype=torch.float32)
    launch_quantize(  # type: ignore[no-untyped-call]
        source,
        quantized,
        scales,
        group_size,
    )
    return quantized, scales


def _validate(source: Any, group_size: int) -> None:
    if source.ndim < 1:
        raise ValueError("source must have at least one dimension")
    if group_size < 16 or group_size > 1024 or group_size & (group_size - 1):
        raise ValueError("group_size must be a power of two in [16, 1024]")
    if source.numel() % group_size:
        raise ValueError("source element count must be divisible by group_size")
