#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_command="${PYTHON:-python}"
games="${DOUZERO_GAMES:-100}"
seed="${MASTER_SEED:-20260722}"

"$python_command" scripts/fetch_douzero_baseline.py
"$python_command" scripts/differential_douzero.py --games "$games" --seed "$seed" "$@"
