from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools import check_source_checkout as checker
from tools.source_versions import expected_worker_versions, load_versions


def run(root: Path, *args: str) -> str:
    return checker.git(root, *args)


def commit(root: Path, message: str = "fixture") -> None:
    run(root, "add", "--all")
    run(root, "commit", "--quiet", "-m", message)


def save_lock(root: Path, data: dict[str, Any]) -> None:
    (root / "versions.lock.yaml").write_text(yaml.safe_dump(data))


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, dict[str, Any], Path]:
    fork = tmp_path / "fork"
    root = tmp_path / "parent"
    for directory in (fork, root):
        directory.mkdir()
        run(directory, "init", "--quiet")
        run(directory, "config", "user.name", "Source fixture")
        run(directory, "config", "user.email", "fixture@example.invalid")
    (fork / "engine.txt").write_text("upstream")
    commit(fork)
    upstream = run(fork, "rev-parse", "HEAD")
    (fork / "engine.txt").write_text("fork")
    commit(fork)
    pin = run(fork, "rev-parse", "HEAD")
    tree = run(fork, "rev-parse", "HEAD^{tree}")
    run(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(fork),
        "third_party/vllm",
    )
    uri = "ssh://fixture.invalid/freechat-vllm.git"
    run(root, "config", "-f", ".gitmodules", "submodule.third_party/vllm.url", uri)
    data = {
        "schema": 1,
        "python": "3.12",
        "gpu_stack": {
            "pytorch": "locked-torch",
            "cuda_runtime": "13.0",
            "triton": "locked-triton",
            "transformers": "locked-transformers",
        },
        "vllm": {
            "fork_repository": uri,
            "fork_commit": pin,
            "upstream_commit": upstream,
            "fork_tree": tree,
            "checkout_path": "third_party/vllm",
            "worker_image_digest": "UNRESOLVED",
        },
    }
    save_lock(root, data)
    commit(root)
    return root, data, fork


def test_clean_checkout_matches_all_source_identities(source: Any) -> None:
    root, data, _ = source
    report = checker.check_checkout(root)
    assert report["accepted"] and all(report["checks"].values())
    assert report["source"]["fork_commit"] == data["vllm"]["fork_commit"]
    assert not report["remote_checked"] and not report["runtime_or_gpu_verified"]


def test_image_expectations_are_read_from_the_same_lock(source: Any) -> None:
    root, data, _ = source
    data["gpu_stack"]["pytorch"] = "different-locked-torch"
    save_lock(root, data)
    assert expected_worker_versions(root)["torch"] == "different-locked-torch"
    assert expected_worker_versions(root)["fork_revision"] == data["vllm"]["fork_commit"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("fork_tree", "0" * 40),
        ("fork_commit", "0" * 40),
        ("fork_repository", "ssh://other.invalid/fork.git"),
    ],
)
def test_changed_lock_is_not_mistaken_for_matching_source(
    source: Any, field: str, value: str
) -> None:
    root, data, _ = source
    data["vllm"][field] = value
    save_lock(root, data)
    if field == "fork_commit":
        # The absent object itself must fail closed, rather than turn into a successful lookup.
        with pytest.raises(subprocess.CalledProcessError):
            checker.check_checkout(root, allow_dirty=True)
    else:
        assert not checker.check_checkout(root, allow_dirty=True)["accepted"]


def test_index_gitlink_cannot_silently_change(source: Any) -> None:
    root, data, _ = source
    run(
        root,
        "update-index",
        "--cacheinfo",
        "160000," + data["vllm"]["upstream_commit"] + ",third_party/vllm",
    )
    report = checker.check_checkout(root, allow_dirty=True)
    assert not report["accepted"] and not report["checks"]["index_gitlink"]


def test_checked_out_fork_must_match_pin(source: Any) -> None:
    root, data, _ = source
    run(root / "third_party/vllm", "checkout", "--quiet", data["vllm"]["upstream_commit"])
    report = checker.check_checkout(root, allow_dirty=True)
    assert not report["accepted"] and not report["checks"]["fork_commit"]


@pytest.mark.parametrize("submodule", [False, True])
def test_dirty_source_requires_explicit_development_override(source: Any, submodule: bool) -> None:
    root, _, _ = source
    directory = root / "third_party/vllm" if submodule else root
    (directory / "uncommitted.txt").write_text("not part of pinned source")
    assert not checker.check_checkout(root)["accepted"]
    assert checker.check_checkout(root, allow_dirty=True)["accepted"]
    report = checker.check_checkout(root, release=True, allow_dirty=True)
    assert not report["accepted"] and not report["dirty_override"]


def test_uninitialized_submodule_is_not_parent_repository(source: Any) -> None:
    root, _, _ = source
    marker = root / "third_party/vllm/.git"
    marker.rename(root.parent / "saved-submodule-git-marker")
    report = checker.check_checkout(root, allow_dirty=True)
    assert not report["accepted"]
    assert not report["checks"]["submodule_initialized"]


def test_release_prerequisite_does_not_silently_accept_unresolved_digest(source: Any) -> None:
    root, data, _ = source
    assert checker.check_checkout(root)["accepted"]
    assert not checker.check_checkout(root, release=True)["accepted"]
    data["vllm"]["worker_image_digest"] = "registry.invalid/worker@sha256:" + "a" * 64
    save_lock(root, data)
    commit(root)
    report = checker.check_checkout(root, release=True)
    assert report["accepted"] and not report["runtime_or_gpu_verified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("fork_commit", "main"),
        ("fork_tree", "invalid"),
        ("checkout_path", "../elsewhere"),
        ("fork_repository", "/unqualified/path"),
    ],
)
def test_invalid_source_identity_fails_validation(source: Any, field: str, value: str) -> None:
    root, data, _ = source
    data["vllm"][field] = value
    save_lock(root, data)
    with pytest.raises(ValueError):
        load_versions(root)


def test_actual_remote_probe_fetches_into_disposable_repository(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, data, fork = source
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url." + str(fork) + ".insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", data["vllm"]["fork_repository"])
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "always")
    before = run(root, "status", "--porcelain=v1")
    report = checker.check_checkout(root, check_remote=True)
    assert report["accepted"] and report["checks"]["remote_tree"]
    assert run(root, "status", "--porcelain=v1") == before


@pytest.mark.parametrize("missing", [False, True])
def test_cli_returns_nonzero_for_invalid_or_missing_checkout(
    source: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: bool
) -> None:
    root, _, _ = source
    if missing:
        root = root / "missing"
    else:
        (root / "unexpected.txt").write_text("dirty")
    monkeypatch.setattr("sys.argv", ["check_source_checkout", "--root", str(root)])
    assert checker.main() == 1
    assert '"accepted": false' in capsys.readouterr().out


def test_malformed_yaml_is_reported_as_a_failed_check(
    source: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _, _ = source
    (root / "versions.lock.yaml").write_text("invalid: [")
    monkeypatch.setattr("sys.argv", ["check_source_checkout", "--root", str(root)])
    assert checker.main() == 1
    assert '"error": "ValueError"' in capsys.readouterr().out


def test_cli_success_does_not_imply_gpu_acceptance(
    source: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _, _ = source
    monkeypatch.setattr("sys.argv", ["check_source_checkout", "--root", str(root)])
    assert checker.main() == 0
    assert '"runtime_or_gpu_verified": false' in capsys.readouterr().out
