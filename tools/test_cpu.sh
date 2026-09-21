#!/usr/bin/env bash
# Partial CPU validation only. GPU, WebUI and fork validation remain separate.
set -euo pipefail
cd "$(dirname "$0")/.."
coverage_data="$(mktemp "${TMPDIR:-/tmp}/freechat-coverage.XXXXXX")"
trap 'rm -f -- "$coverage_data"' EXIT
export COVERAGE_FILE="$coverage_data"

uv run --locked --all-packages pytest -m "not gpu and not multinode" --cov --cov-report=term:skip-covered "$@"
uv run --locked --all-packages python -m tools.check_test_coverage --data-file "$coverage_data"
