"""Validate a pinned source checkout; never deploy or modify working-tree source."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools.source_versions import ROOT, SourcePin, load_versions


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _remote_tree(pin: SourcePin) -> str:
    # Only the disposable local repository is written; the source repository is read-only.
    with tempfile.TemporaryDirectory(prefix="freechat-source-check-") as directory:
        root = Path(directory)
        git(root, "init", "--bare", "--quiet")
        git(root, "fetch", "--quiet", "--no-tags", pin.fork_repository, pin.fork_commit)
        return git(root, "rev-parse", pin.fork_commit + "^{tree}")


def check_checkout(
    root: Path, *, allow_dirty: bool = False, release: bool = False, check_remote: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    lock = load_versions(root)
    pin = lock.vllm
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}
    checks["root_is_repository"] = Path(git(root, "rev-parse", "--show-toplevel")).resolve() == root
    details["parent_commit"] = git(root, "rev-parse", "HEAD")
    path = root / pin.checkout_path
    checks["submodule_initialized"] = (path / ".git").exists()
    record = git(root, "ls-tree", "HEAD", "--", pin.checkout_path).split()
    expected = ["160000", "commit", pin.fork_commit, pin.checkout_path]
    checks["committed_gitlink"] = record == expected
    index = git(root, "ls-files", "--stage", "--", pin.checkout_path).split()
    checks["index_gitlink"] = index == ["160000", pin.fork_commit, "0", pin.checkout_path]
    url = git(
        root, "config", "-f", ".gitmodules", "--get", "submodule." + pin.checkout_path + ".url"
    )
    checks["repository_uri"] = url == pin.fork_repository
    if checks["submodule_initialized"]:
        checks["submodule_is_repository"] = (
            Path(git(path, "rev-parse", "--show-toplevel")).resolve() == path
        )
        details["fork_commit"] = git(path, "rev-parse", "HEAD")
        details["fork_tree"] = git(path, "rev-parse", "HEAD^{tree}")
        checks["fork_commit"] = details["fork_commit"] == pin.fork_commit
        checks["fork_tree"] = details["fork_tree"] == pin.fork_tree
        checks["upstream_ancestor"] = (
            git(path, "merge-base", pin.upstream_commit, pin.fork_commit) == pin.upstream_commit
        )
    else:
        checks["fork_commit"] = checks["fork_tree"] = checks["upstream_ancestor"] = False
    dirty = bool(
        git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    )
    checks["clean_source"] = not dirty
    if release:
        checks["immutable_worker_image"] = (
            re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-f]{64}", pin.worker_image_digest) is not None
        )
    if check_remote:
        checks["remote_tree"] = _remote_tree(pin) == pin.fork_tree
    required = {
        key: value
        for key, value in checks.items()
        if key != "clean_source" or not allow_dirty or release
    }
    return {
        "scope": "SOURCE_RELEASE_PREREQUISITE" if release else "SOURCE_CONSISTENCY",
        "accepted": all(required.values()),
        "dirty_override": allow_dirty and not release,
        "remote_checked": check_remote,
        "checks": checks,
        "source": details,
        "worker_image_digest": pin.worker_image_digest,
        "runtime_or_gpu_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-dirty", action="store_true", help="Development only")
    parser.add_argument(
        "--release",
        action="store_true",
        help="Also require an immutable image digest; not runtime acceptance",
    )
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="Fetch the pin into a disposable local bare repository",
    )
    args = parser.parse_args()
    try:
        report = check_checkout(
            args.root,
            allow_dirty=args.allow_dirty,
            release=args.release,
            check_remote=args.check_remote,
        )
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        report = {
            "accepted": False,
            "error": type(error).__name__,
            "runtime_or_gpu_verified": False,
        }
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
