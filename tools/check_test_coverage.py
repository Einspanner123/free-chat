"""Enforce owned Python statement and branch coverage; print results only.

This gate includes unimported runtime, GPU and validation modules. It does not
measure WebUI or the independent vLLM fork, and is not GPU execution acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

from coverage import Coverage
from coverage.exceptions import CoverageException

SOURCE_ROOTS = ("libs", "services", "worker", "tools", "benchmarks")
MINIMUM_PERCENT = 80
_CACHE_PARTS = frozenset({"__pycache__", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
ROOT = Path(__file__).resolve().parents[1]


def production_files(root: Path) -> tuple[Path, ...]:
    """Discover current source, including untracked and never-imported modules."""
    root = root.resolve()
    files: list[Path] = []
    for name in SOURCE_ROOTS:
        directory = root / name
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"missing_or_linked_source_root:{name}")
        found: list[Path] = []
        for path in directory.rglob("*"):
            relative = path.relative_to(directory)
            if "tests" in relative.parts or _CACHE_PARTS.intersection(relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"linked_source_not_supported:{path.relative_to(root)}")
            if not path.is_file() or path.suffix != ".py" or path.name.startswith("test_"):
                continue
            # Only compiler-produced protocol modules are outside this denominator.
            if (
                name == "libs"
                and relative.parts[:2] == ("protocol", "src")
                and (path.name.endswith("_pb2.py") or path.name.endswith("_pb2_grpc.py"))
            ):
                continue
            found.append(path)
        if not found:
            raise ValueError(f"empty_source_root:{name}")
        files.extend(found)
    return tuple(sorted(files))


def _percent(covered: int, total: int) -> float:
    return 100.0 if total == 0 else 100.0 * covered / total


def _counts(summary: dict[str, Any]) -> dict[str, int | float]:
    lines, statements = int(summary["covered_lines"]), int(summary["num_statements"])
    branches, exits = int(summary["covered_branches"]), int(summary["num_branches"])
    return {
        "covered_statements": lines,
        "statements": statements,
        "statement_percent": _percent(lines, statements),
        "covered_branches": branches,
        "branches": exits,
        "branch_percent": _percent(branches, exits),
    }


def check_coverage(root: Path, data_file: Path) -> dict[str, Any]:
    root = root.resolve()
    files = production_files(root)
    if not data_file.is_file():
        raise ValueError("coverage_data_missing")
    # Do not inherit a caller's omit/include/fail-under overrides. Explicit morfs
    # prevents coverage.py's import/discovery heuristics from shrinking the source set.
    coverage = Coverage(data_file=str(data_file), config_file=False, branch=True)
    coverage.load()
    if not coverage.get_data().has_arcs():
        raise ValueError("branch_coverage_data_required")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        coverage.json_report(morfs=[str(path) for path in files], outfile="-")
    report = json.loads(output.getvalue())
    reported = {Path(name).resolve(): value for name, value in report["files"].items()}
    if set(reported) != set(files):
        raise ValueError("coverage_source_set_mismatch")
    totals = _counts(report["totals"])
    if not totals["statements"]:
        raise ValueError("empty_statement_denominator")
    for field in ("covered_lines", "num_statements", "covered_branches", "num_branches"):
        if sum(entry["summary"][field] for entry in reported.values()) != report["totals"][field]:
            raise ValueError("coverage_denominator_mismatch")
    low_coverage = []
    for path, entry in sorted(reported.items()):
        counts = _counts(entry["summary"])
        if (
            counts["statement_percent"] < MINIMUM_PERCENT
            or counts["branch_percent"] < MINIMUM_PERCENT
        ):
            low_coverage.append({"file": str(path.relative_to(root)), **counts})
    return {
        "scope": "OWNED_PYTHON_TEST_COVERAGE",
        "accepted": (
            totals["statement_percent"] >= MINIMUM_PERCENT
            and totals["branch_percent"] >= MINIMUM_PERCENT
        ),
        "minimum_statement_percent": MINIMUM_PERCENT,
        "minimum_branch_percent": MINIMUM_PERCENT,
        "source_roots": list(SOURCE_ROOTS),
        "source_files": len(files),
        "source_set_verified": True,
        "totals": totals,
        "below_threshold_files": low_coverage,
        "webui_and_vllm_fork_included": False,
        "gpu_execution_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = check_coverage(args.root, args.data_file)
    except (ValueError, OSError, CoverageException) as error:
        report = {
            "scope": "OWNED_PYTHON_TEST_COVERAGE",
            "accepted": False,
            "error": str(error),
            "gpu_execution_verified": False,
        }
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
