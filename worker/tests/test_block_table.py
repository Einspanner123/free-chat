import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="GPU dependency group is not installed")
def test_reference_gathers_selected_rows() -> None:
    import torch  # type: ignore[import-not-found]
    from freechat_worker.kernels import gather_block_rows_reference

    table = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    rows = torch.tensor([3, 1, 1], dtype=torch.int64)
    assert torch.equal(gather_block_rows_reference(table, rows), table[[3, 1, 1]])


@pytest.mark.gpu
@pytest.mark.skipif(not torch_available, reason="GPU dependency group is not installed")
def test_triton_gather_matches_reference_on_cuda() -> None:
    import torch
    from freechat_worker.kernels import gather_block_rows, gather_block_rows_reference

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7)
    table = torch.randint(0, 100_000, (513, 257), dtype=torch.int32, device="cuda")
    rows = torch.randint(0, 513, (37,), dtype=torch.int32, device="cuda", generator=generator)
    actual = gather_block_rows(table, rows)
    expected = gather_block_rows_reference(table, rows)
    assert torch.equal(actual, expected)
