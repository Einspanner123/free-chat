import json
from pathlib import Path

import pytest
from freechat_scheduler.local_cluster import main_async


async def test_cli_emits_explicitly_synthetic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "nested" / "layouts.json"
    monkeypatch.setattr("sys.argv", ["local-cluster", "--output", str(output)])
    await main_async()
    result = json.loads(output.read_text())
    assert "no model execution" in result["scope"]
    assert [layout["resource_groups"] for layout in result["layouts"]] == [12, 6, 3]
    for layout in result["layouts"]:
        assert layout["hardware_verified"] is False
        assert layout["evidence_level"] == "LOCAL_SIMULATION_ONLY"
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        await main_async()
    assert error.value.code == 2
    assert output.read_bytes() == original
