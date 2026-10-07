#!/usr/bin/env bash
# Runs the same checks as CI: Ruff, then the test suite.
#
# The tools come from the .venv in the main checkout, which every git
# worktree of the clone shares, so a fresh worktree needs no install of its
# own. Without that .venv, fall back to whatever Python is on PATH.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
main_root="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"

python=""
for candidate in \
    "$main_root/.venv/Scripts/python.exe" \
    "$main_root/.venv/bin/python" \
    "$repo_root/.venv/Scripts/python.exe" \
    "$repo_root/.venv/bin/python"; do
    if [ -x "$candidate" ]; then
        python="$candidate"
        break
    fi
done
if [ -z "$python" ]; then
    python="$(command -v python3 || command -v python)"
fi

cd "$repo_root"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

"$python" -m ruff check .
"$python" -m pytest -q
