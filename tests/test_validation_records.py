from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from benchmarks import analyze_offload_boundary as analysis
from benchmarks.records import emit_record


def test_records_and_analysis_use_stdout_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = "[]"
    reference = emit_record("index.json", payload)
    line = capsys.readouterr().out
    record = json.loads(line)
    assert reference == {
        "name": "index.json",
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    assert record["payload"] == payload
    monkeypatch.setattr("sys.argv", ["analyze"])
    monkeypatch.setattr("sys.stdin", io.StringIO(line))
    analysis.main()
    report = json.loads(capsys.readouterr().out)
    assert report["groups"] == {}
    assert report["final_ab_acceptance"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["hash", "duplicate", "missing_index"])
def test_stream_rejects_corruption_or_incomplete_records(
    failure: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    emit_record("index.json", "[]")
    line = capsys.readouterr().out
    expected: type[Exception] = ValueError
    if failure == "hash":
        record = json.loads(line)
        record["payload"] = "tampered"
        line = json.dumps(record) + "\n"
    elif failure == "duplicate":
        line += line
    else:
        line, expected = "", KeyError
    monkeypatch.setattr("sys.argv", ["analyze"])
    monkeypatch.setattr("sys.stdin", io.StringIO(line))
    with pytest.raises(expected):
        analysis.main()
