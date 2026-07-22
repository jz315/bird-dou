#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_command="${PYTHON:-python}"
games="${BELIEF_GAMES:-100}"
seed="${MASTER_SEED:-5005}"
dataset="${BELIEF_DATASET:-artifacts/datasets/belief_smoke.npz}"
report="${BELIEF_REPORT:-artifacts/train/belief_pretrain/report.json}"

"$python_command" -m birddou.cli.generate_dataset \
  --games "$games" --seed "$seed" --output "$dataset"
"$python_command" -m birddou.cli.train_belief \
  --config "${BELIEF_CONFIG:-configs/train/belief_pretrain.yaml}" \
  --report "$report" "$@"
