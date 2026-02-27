#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/run_formal_a10.sh
#   WORKERS=16 BATCH_SIZE=512 LR=0.002 bash tools/run_formal_a10.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-formal_a10_loco_h10_bs384_lr1p6e3}"
CACHE_DIR="${CACHE_DIR:-datasets/sound_api/cache_npz}"
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-384}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-0.0016}"
HORIZON="${HORIZON:-10}"

mkdir -p experiments/logs
LOG_FILE="experiments/logs/${RUN_NAME}.log"

echo "Starting training..."
echo "run_name=$RUN_NAME"
echo "log_file=$LOG_FILE"

CMD=(
  python experiments/train.py
  --data_source sound_api_cache
  --task risk
  --horizon "$HORIZON"
  --cache_dir "$CACHE_DIR"
  --device cuda
  --amp
  --workers "$WORKERS"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --sound_split_mode leave_one_condition_out
  --condition_policy xjtu_3cond
  --calibrate_threshold
  --optimizer adamw
  --weight_decay 2e-4
  --scheduler cosine
  --min_lr 5e-7
  --early_stop_patience 10
  --run_name "$RUN_NAME"
)

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
