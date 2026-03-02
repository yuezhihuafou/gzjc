#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

THRESHOLDS="${THRESHOLDS:-0.5 0.9 0.95}"

CKPT_DIR="$(find experiments/runs -type f -name backbone.pth -printf '%T@ %h\n' | sort -n | tail -n 1 | awk '{print $2}')"
if [[ -z "${CKPT_DIR}" ]]; then
  echo "未找到 checkpoint（experiments/runs/*/checkpoints/backbone.pth）"
  exit 1
fi

CWRU_DIR="$(find datasets/cwru -type f -name signals.npy -printf '%h\n' | head -n 1)"
if [[ -z "${CWRU_DIR}" ]]; then
  echo "未找到 CWRU signals.npy（datasets/cwru/**/signals.npy）"
  exit 1
fi

echo "use_ckpt=${CKPT_DIR}"
echo "use_cwru=${CWRU_DIR}"

for th in ${THRESHOLDS}; do
  echo "=== eval threshold=${th} ==="
  python tools/eval_cross_domain.py \
    --checkpoint_dir "${CKPT_DIR}" \
    --cwru_dir "${CWRU_DIR}" \
    --device cuda \
    --threshold "${th}"
done

echo "done"
