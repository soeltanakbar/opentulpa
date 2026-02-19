#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

# Ensure we're in the right directory
cd "${REPO_ROOT}"

# Launch the Python manager
exec uv run python scripts/manager.py "$@"
