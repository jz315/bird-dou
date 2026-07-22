#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_command="${PYTHON:-python}"

"$python_command" -m birddou.cli.crossplay \
  --output "${CROSSPLAY_REPORT:-artifacts/eval/crossplay.json}" \
  "$@"
