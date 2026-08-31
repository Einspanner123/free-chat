# mypy: ignore-errors

import triton  # type: ignore[import-not-found]
import triton.language as tl  # type: ignore[import-not-found]


@triton.jit
def _gather_kernel(source, rows, output, width, output_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    output_row = offsets // width
    column = offsets % width
    source_row = tl.load(rows + output_row, mask=mask, other=0)
    values = tl.load(source + source_row * width + column, mask=mask)
    tl.store(output + offsets, values, mask=mask)


def launch_gather(source, rows, output):
    output_size = output.numel()
    block_size = 256
    grid = (triton.cdiv(output_size, block_size),)
    _gather_kernel[grid](
        source,
        rows,
        output,
        source.shape[1],
        output_size,
        BLOCK_SIZE=block_size,
    )
