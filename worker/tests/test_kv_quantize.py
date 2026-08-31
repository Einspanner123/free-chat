import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.gpu
@pytest.mark.skipif(not torch_available, reason="GPU dependency group is not installed")
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("group_size", [32, 128, 256])
def test_triton_kv_quantize_matches_reference(dtype_name: str, group_size: int) -> None:
    import torch  # type: ignore[import-not-found]
    from freechat_worker.kernels import quantize_kv, quantize_kv_reference

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = getattr(torch, dtype_name)
    source = torch.randn(4096, device="cuda", dtype=dtype)
    actual_values, actual_scales = quantize_kv(
        source,
        group_size,
        enable_triton=True,
    )
    expected_values, expected_scales = quantize_kv_reference(source, group_size)
    integer_error = (actual_values.to(torch.int16) - expected_values.to(torch.int16)).abs()
    assert integer_error.max().item() <= 1
    assert torch.allclose(actual_scales, expected_scales, rtol=1e-5, atol=1e-7)
    restored = actual_values.reshape(-1, group_size).float() * actual_scales[:, None]
    reconstruction_error = (restored.reshape_as(source).float() - source.float()).abs()
    quantization_bound = actual_scales.repeat_interleave(group_size) / 2 + 1e-5
    assert torch.all(reconstruction_error <= quantization_bound)


@pytest.mark.skipif(not torch_available, reason="GPU dependency group is not installed")
def test_reference_quantization_has_bounded_reconstruction_error() -> None:
    import torch
    from freechat_worker.kernels import quantize_kv_reference

    source = torch.linspace(-3, 3, 1024, dtype=torch.float32)
    values, scales = quantize_kv_reference(source, group_size=128)
    restored = values.reshape(-1, 128).float() * scales[:, None]
    error = (restored.reshape_as(source) - source).abs()
    assert torch.all(error <= scales.repeat_interleave(128) / 2 + 1e-6)
