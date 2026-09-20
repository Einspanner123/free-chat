import json
from pathlib import Path

import pytest
from freechat_scheduler.local_cluster import main_async


async def test_cli_emits_synthetic_evidence_to_stdout_without_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["local-cluster"])
    await main_async()
    result = json.loads(capsys.readouterr().out)
    assert "no model execution" in result["scope"]
    assert [layout["resource_groups"] for layout in result["layouts"]] == [12, 6, 3]
    for layout in result["layouts"]:
        assert layout["hardware_verified"] is False
        assert layout["evidence_level"] == "LOCAL_SIMULATION_ONLY"
    assert not list(tmp_path.iterdir())
