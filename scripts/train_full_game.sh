#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_command="${PYTHON:-python}"

"$python_command" -m birddou.cli.train_full_game \
  --config "${FULL_GAME_CONFIG:-configs/train/full_game_smoke.yaml}" \
  --report "${FULL_GAME_REPORT:-artifacts/train/full_game_smoke/report.json}" \
  "$@"
