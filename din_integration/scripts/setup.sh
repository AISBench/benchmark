#!/bin/bash
# ============================================================================
# setup.sh — SWE-bench DinD host-side setup helper
# ============================================================================
# Creates the runtime directories and symlinks expected by the
# swebench-dind CLI. Idempotent.
#
# Usage:
#   bash din_integration/scripts/setup.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[setup] creating host runtime directories under \$HOME..."
mkdir -p "$HOME/swebench_dind_jobs"
mkdir -p "$HOME/swebench_dind_tasks"
mkdir -p "$HOME/swebench_dind_logs"
mkdir -p "$HOME/.config/swebench-dind"

echo "[setup] expected api_key.env location: $HOME/.config/swebench-dind/api_key.env"
if [[ ! -f "$HOME/.config/swebench-dind/api_key.env" ]]; then
    echo "[setup] WARNING: $HOME/.config/swebench-dind/api_key.env not found."
    echo "         Copy $ROOT/scripts/api_key.env.template to that location and edit."
fi

echo "[setup] checking legacy launcher.py path expectation..."
LEGACY_DIR="/home/zengziyu/mini_matrix/scripts"
if [[ -d "/home/zengziyu/mini_matrix" ]]; then
    if [[ ! -f "$LEGACY_DIR/api_key.env" ]]; then
        echo "[setup] WARNING: launcher.py currently hardcodes $LEGACY_DIR/api_key.env"
        echo "         Either copy your key there, or wait for env-var refactor."
    fi
fi

echo "[setup] done. Next:"
echo "         cd $ROOT && pip install -e ."
echo "         bash $ROOT/scripts/start_orchestrator.sh"