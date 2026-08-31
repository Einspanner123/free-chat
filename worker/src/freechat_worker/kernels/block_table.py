from __future__ import annotations

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


def gather_block_rows(block_table: Any, row_indices: Any) -> Any:
    """Gather active branch rows with a Triton kernel and safe fallback."""
    import torch

    if not block_table.is_cuda or not row_indices.is_cuda:
        return gather_block_rows_reference(block_table, row_indices)
    if block_table.dtype != torch.int32 or row_indices.dtype != torch.int32:
        return gather_block_rows_reference(block_table, row_indices)
    if not block_table.is_contiguous() or not row_indices.is_contiguous():
        return gather_block_rows_reference(block_table, row_indices)

    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    @triton.jit  # type: ignore[untyped-decorator]
    def gather_kernel(
        source: Any,
        rows: Any,
        output: Any,
        width: int,
        output_size: int,
        block_size: tl.constexpr,
    ) -> None:
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < output_size
        output_row = offsets // width
        column = offsets % width
        source_row = tl.load(rows + output_row, mask=mask, other=0)
        values = tl.load(source + source_row * width + column, mask=mask)
        tl.store(output + offsets, values, mask=mask)

    output = torch.empty(
        (row_indices.numel(), block_table.shape[1]),
        device=block_table.device,
        dtype=block_table.dtype,
    )
    output_size = output.numel()
    if output_size:
        block_size = 256
        grid = (triton.cdiv(output_size, block_size),)
        gather_kernel[grid](
            block_table,
            row_indices,
            output,
            block_table.shape[1],
            output_size,
            block_size=block_size,
        )
    return output
