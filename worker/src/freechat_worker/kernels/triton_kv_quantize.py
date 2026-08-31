# mypy: ignore-errors

import triton  # type: ignore[import-not-found]
import triton.language as tl  # type: ignore[import-not-found]
from triton.language.extra import libdevice  # type: ignore[import-not-found]


@triton.jit
def _quantize_kernel(source, quantized, scales, GROUP_SIZE: tl.constexpr):
    group = tl.program_id(0)
    offsets = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    values = tl.load(source + offsets).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(maximum / 127.0, 1e-12)
    packed = libdevice.rint(values / scale)
    packed = tl.maximum(-127.0, tl.minimum(127.0, packed))
    tl.store(quantized + offsets, packed.to(tl.int8))
    tl.store(scales + group, scale)


def launch_quantize(source, quantized, scales, group_size):
    grid = (source.numel() // group_size,)
    _quantize_kernel[grid](
        source,
        quantized,
        scales,
        GROUP_SIZE=group_size,
        num_warps=4,
    )
