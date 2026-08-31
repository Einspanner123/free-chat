from unittest.mock import patch

from freechat_topology_agent.collector import _collect_gpus


def test_gpu_csv_is_parsed_without_shell() -> None:
    row = "0, GPU-id, NVIDIA RTX A6000, 580.173.02, 49140, 8.6"
    with patch("freechat_topology_agent.collector._run_optional", return_value=row):
        devices = _collect_gpus()
    assert devices[0].name == "NVIDIA RTX A6000"
    assert devices[0].memory_total_bytes == 49_140 * 1024**2
