from __future__ import annotations

import os
from typing import Any


def gather_block_rows_reference(block_table: Any, row_indices: Any) -> Any:
    """Reference branch-resume block-table gather.

    Imports stay local so the CPU control-plane environment does not require
    the GPU dependency group.
    """
    import torch  # type: ignore[import-not-found]

    if block_table.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("block_table must be 2-D and row_indices must be 1-D")
    return torch.index_select(block_table, 0, row_indices.to(dtype=torch.int64))


def gather_block_rows(
    block_table: Any,
    row_indices: Any,
    *,
    enable_triton: bool | None = None,
) -> Any:
    """Gather active branch rows with a Triton kernel and safe fallback."""
    import torch

    enabled = (
        os.environ.get("FREECHAT_ENABLE_TRITON_BLOCK_TABLE") == "1"
        if enable_triton is None
        else enable_triton
    )
    if not enabled or not block_table.is_cuda or not row_indices.is_cuda:
        return gather_block_rows_reference(block_table, row_indices)
    if block_table.dtype != torch.int32 or row_indices.dtype != torch.int32:
        return gather_block_rows_reference(block_table, row_indices)
    if not block_table.is_contiguous() or not row_indices.is_contiguous():
        return gather_block_rows_reference(block_table, row_indices)

    from freechat_worker.kernels.triton_block_table import launch_gather

    output = torch.empty(
        (row_indices.numel(), block_table.shape[1]),
        device=block_table.device,
        dtype=block_table.dtype,
    )
    output_size = output.numel()
    if output_size:
        launch_gather(block_table, row_indices, output)  # type: ignore[no-untyped-call]
    return output
