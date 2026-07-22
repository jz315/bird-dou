#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_command="${PYTHON:-python}"

"$python_command" -m birddou.cli.train \
  --config "${CARDPLAY_CONFIG:-configs/train/bird_dou_dmc_smoke.yaml}" \
  --report "${CARDPLAY_REPORT:-artifacts/train/bird_dou_dmc_smoke/report.json}" \
  "$@"
