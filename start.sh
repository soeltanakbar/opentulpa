#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

# Ensure we're in the right directory
cd "${REPO_ROOT}"

# Observability defaults: bring up OpenLIT/OTLP stack alongside the app unless explicitly disabled.
export OPENLIT_ENABLED="${OPENLIT_ENABLED:-true}"
export OPENLIT_AUTO_START="${OPENLIT_AUTO_START:-true}"

# Launch the Python manager
exec uv run python scripts/manager.py "$@"
