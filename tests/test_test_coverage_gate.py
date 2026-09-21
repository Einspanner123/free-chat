from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from coverage import Coverage, CoverageData

from tools import check_test_coverage as gate


@pytest.fixture
def source(tmp_path: Path) -> Path:
    for name in gate.SOURCE_ROOTS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "module.py").write_text("value = 1\n")
    return tmp_path


def data_file(root: Path, *, omit: str | None = None, branch: bool = True) -> Path:
    """Create tiny public CoverageData fixtures, not runtime/GPU evidence."""
    path = root / ".test-coverage"
    data = CoverageData(basename=str(path))
    files = [p for p in gate.production_files(root) if p.parent.name != omit]
    if branch:
        data.add_arcs({str(p): [(-1, 1), (1, -1)] for p in files})
    else:
        data.add_lines({str(p): [1] for p in files})
    data.write()
    return path


def test_complete_statement_and_branch_gate(source: Path) -> None:
    report = gate.check_coverage(source, data_file(source))
    assert report["accepted"]
    assert report["source_files"] == 5
    assert report["totals"]["statement_percent"] == 100
    assert report["totals"]["branch_percent"] == 100
    assert report["below_threshold_files"] == []
    assert not report["gpu_execution_verified"]
    assert not report["webui_and_vllm_fork_included"]


def test_unimported_file_is_zero_and_stays_in_denominator(source: Path) -> None:
    report = gate.check_coverage(source, data_file(source, omit="worker"))
    assert report["accepted"]  # Exactly four of five statements, not five of five.
    assert report["totals"]["statement_percent"] == 80
    assert report["below_threshold_files"][0]["file"] == "worker/module.py"
    assert report["below_threshold_files"][0]["covered_statements"] == 0


def test_new_untracked_runtime_and_gpu_modules_are_not_dropped(source: Path) -> None:
    recorded = data_file(source)
    for name in ("serve.py", "kernels/quantize.py"):
        path = source / "worker" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("untested = 1\n")
    report = gate.check_coverage(source, recorded)
    assert not report["accepted"]
    assert report["source_files"] == 7
    assert report["totals"]["statements"] == 7
    assert {row["file"] for row in report["below_threshold_files"]} == {
        "worker/serve.py",
        "worker/kernels/quantize.py",
    }


@pytest.mark.parametrize(
    "path",
    [
        "worker/tests/test_worker.py",
        "worker/test_fixture.py",
        "worker/__pycache__/cached.py",
        "services/.venv/external.py",
        "tools/.pytest_cache/cached.py",
        "libs/protocol/src/freechat/control/v1/control_pb2.py",
        "libs/protocol/src/freechat/control/v1/control_pb2_grpc.py",
        "third_party/vllm/engine.py",
    ],
)
def test_only_tests_generated_cache_and_outside_scope_are_excluded(source: Path, path: str) -> None:
    candidate = source / path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("not_in_denominator = 1\n")
    assert len(gate.production_files(source)) == 5


def test_pb2_suffix_outside_protocol_tree_is_not_a_blanket_exclusion(source: Path) -> None:
    (source / "tools" / "own_pb2.py").write_text("owned_code = 1\n")
    assert len(gate.production_files(source)) == 6


@pytest.mark.parametrize("missing", gate.SOURCE_ROOTS)
def test_missing_source_root_fails_closed(source: Path, missing: str) -> None:
    (source / missing).rename(source / f"saved-{missing}")
    with pytest.raises(ValueError, match="missing_or_linked_source_root"):
        gate.production_files(source)


def test_empty_source_root_fails_closed(source: Path) -> None:
    (source / "tools" / "module.py").unlink()
    with pytest.raises(ValueError, match="empty_source_root"):
        gate.production_files(source)


@pytest.mark.parametrize("directory", [False, True])
def test_source_symlink_cannot_hide_unmeasured_code(source: Path, directory: bool) -> None:
    target = source / "services" if directory else source / "services" / "module.py"
    (source / "tools" / "alias").symlink_to(target, target_is_directory=directory)
    with pytest.raises(ValueError, match="linked_source_not_supported"):
        gate.production_files(source)


def test_source_root_symlink_is_rejected(source: Path) -> None:
    (source / "tools").rename(source / "saved-tools")
    (source / "tools").symlink_to(source / "saved-tools", target_is_directory=True)
    with pytest.raises(ValueError, match="missing_or_linked_source_root"):
        gate.production_files(source)


def test_missing_data_never_passes(source: Path) -> None:
    with pytest.raises(ValueError, match="coverage_data_missing"):
        gate.check_coverage(source, source / "absent")


def test_statement_only_data_never_passes_branch_gate(source: Path) -> None:
    with pytest.raises(ValueError, match="branch_coverage_data_required"):
        gate.check_coverage(source, data_file(source, branch=False))


def test_branch_threshold_is_independent_from_statement_threshold(source: Path) -> None:
    path = source / "worker" / "module.py"
    path.write_text("if flag:\n    value = 1\nelse:\n    value = 2\n")
    recorded = data_file(source)
    data = CoverageData(basename=str(recorded))
    data.read()
    # Every line was visited, but both branch arcs were intentionally not recorded.
    data.add_arcs({str(path): [(1, 2), (2, 4), (4, -1)]})
    data.write()
    report = gate.check_coverage(source, recorded)
    assert report["totals"]["statement_percent"] == 100
    assert report["totals"]["branch_percent"] < 80
    assert not report["accepted"]


def test_empty_statement_denominator_does_not_pass(source: Path) -> None:
    recorded = data_file(source)
    for path in gate.production_files(source):
        path.write_text("# no executable code\n")
    with pytest.raises(ValueError, match="empty_statement_denominator"):
        gate.check_coverage(source, recorded)


@pytest.mark.parametrize("mismatch", ["files", "totals"])
def test_report_source_and_denominator_must_match(
    source: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    recorded = data_file(source)
    summary = {
        "covered_lines": 1,
        "num_statements": 1,
        "covered_branches": 0,
        "num_branches": 0,
    }
    entries = {str(p): {"summary": summary} for p in gate.production_files(source)}
    if mismatch == "files":
        entries.pop(next(iter(entries)))
    payload = {"files": entries, "totals": {**summary, "covered_lines": 6, "num_statements": 6}}

    def fake_report(*args: Any, **kwargs: Any) -> float:
        print(json.dumps(payload))
        return 100

    monkeypatch.setattr(Coverage, "json_report", fake_report)
    reason = "source_set_mismatch" if mismatch == "files" else "denominator_mismatch"
    with pytest.raises(ValueError, match=reason):
        gate.check_coverage(source, recorded)


@pytest.mark.parametrize("missing", [False, True])
def test_cli_prints_stdout_and_sets_exit_status(
    source: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: bool
) -> None:
    recorded = source / "missing" if missing else data_file(source)
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(source), "--data-file", str(recorded)])
    assert gate.main() == int(missing)
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"] is not missing
    assert not report["gpu_execution_verified"]


def test_near_threshold_is_not_rounded_up() -> None:
    assert gate._percent(7999, 10000) < gate.MINIMUM_PERCENT
