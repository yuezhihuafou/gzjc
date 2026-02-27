#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="all"                        # tune | formal | all
USE_TMUX=0                        # 0 | 1
SESSION_NAME="gzjc-risk"
DATA_SOURCE="sound_api_cache"
CACHE_DIR="datasets/sound_api/cache_npz"
DEVICE="cuda"
WORKERS=6
BATCH_SIZE=128
EPOCHS_TUNE=30
EPOCHS_FORMAL=50
HORIZON=10
LR=0.001
RUNS_ROOT="experiments/runs"
LOGS_ROOT="experiments/logs"
CONDITION_POLICY="xjtu_3cond"
SOUND_SPLIT_MODE="leave_one_condition_out"
MODEL_SCALE="base"
EMBEDDING_DIM="512"
HEAD_DROPOUT="0.0"
INNER_MODE=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_risk_suite.sh [options]

Options:
  --mode <tune|formal|all>      Run tuning set, formal set, or both (default: all)
  --tmux                        Run in tmux detached session
  --session <name>              Tmux session name (default: gzjc-risk)
  --cache_dir <path>            NPZ cache dir (default: datasets/sound_api/cache_npz)
  --runs_root <path>            Archive root for run outputs (default: experiments/runs)
  --logs_root <path>            Log root (default: experiments/logs)
  --device <cuda|cpu>           Device (default: cuda)
  --workers <n>                 DataLoader workers (default: 6)
  --batch_size <n>              Base batch size for non-override runs (default: 128)
  --horizon <n>                 Risk horizon (default: 10)
  --lr <float>                  Base learning rate (default: 1e-3)
  --epochs_tune <n>             Epochs for tuning runs (default: 30)
  --epochs_formal <n>           Epochs for formal runs (default: 50)
  --split_mode <bearing|leave_one_condition_out>
                                sound_api_cache split mode (default: leave_one_condition_out)
  --condition_policy <xjtu_3cond|none>
                                Condition infer policy (default: xjtu_3cond)
  --model_scale <base|large|xlarge>
                                Model scale (default: base)
  --embedding_dim <n>           Embedding dim (default: 512 for base, 768 for large)
  --head_dropout <float>        Head dropout ratio (default: 0.0)
  -h, --help                    Show this help

Examples:
  bash tools/run_risk_suite.sh --mode tune --tmux --session gzjc-tune
  bash tools/run_risk_suite.sh --mode formal --batch_size 128 --epochs_formal 60
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --tmux) USE_TMUX=1; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --cache_dir) CACHE_DIR="$2"; shift 2 ;;
    --runs_root) RUNS_ROOT="$2"; shift 2 ;;
    --logs_root) LOGS_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --horizon) HORIZON="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --epochs_tune) EPOCHS_TUNE="$2"; shift 2 ;;
    --epochs_formal) EPOCHS_FORMAL="$2"; shift 2 ;;
    --split_mode) SOUND_SPLIT_MODE="$2"; shift 2 ;;
    --condition_policy) CONDITION_POLICY="$2"; shift 2 ;;
    --model_scale) MODEL_SCALE="$2"; shift 2 ;;
    --embedding_dim) EMBEDDING_DIM="$2"; shift 2 ;;
    --head_dropout) HEAD_DROPOUT="$2"; shift 2 ;;
    --inner) INNER_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "tune" && "$MODE" != "formal" && "$MODE" != "all" ]]; then
  echo "Invalid --mode: $MODE"
  exit 1
fi

mkdir -p "$RUNS_ROOT" "$LOGS_ROOT"

if [[ $USE_TMUX -eq 1 && $INNER_MODE -eq 0 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Install tmux first."
    exit 1
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    echo "attach: tmux attach -t $SESSION_NAME"
    exit 1
  fi
  CMD=(
    "bash" "tools/run_risk_suite.sh"
    "--mode" "$MODE"
    "--cache_dir" "$CACHE_DIR"
    "--runs_root" "$RUNS_ROOT"
    "--logs_root" "$LOGS_ROOT"
    "--device" "$DEVICE"
    "--workers" "$WORKERS"
    "--batch_size" "$BATCH_SIZE"
    "--horizon" "$HORIZON"
    "--lr" "$LR"
    "--epochs_tune" "$EPOCHS_TUNE"
    "--epochs_formal" "$EPOCHS_FORMAL"
    "--split_mode" "$SOUND_SPLIT_MODE"
    "--condition_policy" "$CONDITION_POLICY"
    "--model_scale" "$MODEL_SCALE"
    "--embedding_dim" "$EMBEDDING_DIM"
    "--head_dropout" "$HEAD_DROPOUT"
    "--inner"
  )
  CMD_STR=""
  for arg in "${CMD[@]}"; do
    CMD_STR+=$(printf "%q " "$arg")
  done
  tmux new-session -d -s "$SESSION_NAME" "cd \"$ROOT_DIR\" && $CMD_STR"
  echo "Started in tmux session: $SESSION_NAME"
  echo "attach: tmux attach -t $SESSION_NAME"
  echo "logs:   $LOGS_ROOT"
  exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="$LOGS_ROOT/risk_suite_$STAMP"
mkdir -p "$SUITE_DIR"
SUMMARY_CSV="$SUITE_DIR/summary.csv"
echo "run_name,status,test_loss,test_acc,test_auc,test_pr_auc,metrics_path,log_path" > "$SUMMARY_CSV"

COMMON_ARGS=(
  --data_source "$DATA_SOURCE"
  --task risk
  --horizon "$HORIZON"
  --cache_dir "$CACHE_DIR"
  --device "$DEVICE"
  --amp
  --workers "$WORKERS"
  --model_scale "$MODEL_SCALE"
  --embedding_dim "$EMBEDDING_DIM"
  --head_dropout "$HEAD_DROPOUT"
  --sound_split_mode "$SOUND_SPLIT_MODE"
  --condition_policy "$CONDITION_POLICY"
  --calibrate_threshold
  --optimizer adamw
  --scheduler cosine
  --min_lr 1e-6
  --early_stop_patience 8
  --runs_root "$RUNS_ROOT"
)

get_metric() {
  local metrics_file="$1"
  local key="$2"
  if [[ ! -f "$metrics_file" ]]; then
    echo ""
    return 0
  fi
  awk -F',' -v k="$key" '$1==k {print $2}' "$metrics_file" | tail -n 1
}

append_summary() {
  local run_name="$1"
  local status="$2"
  local log_file="$3"
  local metrics_file="$RUNS_ROOT/$run_name/outputs/metrics.csv"
  local test_loss test_acc test_auc test_pr_auc
  test_loss="$(get_metric "$metrics_file" test_loss)"
  test_acc="$(get_metric "$metrics_file" test_acc)"
  test_auc="$(get_metric "$metrics_file" test_auc)"
  test_pr_auc="$(get_metric "$metrics_file" test_pr_auc)"
  echo "${run_name},${status},${test_loss},${test_acc},${test_auc},${test_pr_auc},${metrics_file},${log_file}" >> "$SUMMARY_CSV"
}

run_one() {
  local run_name="$1"
  local epochs="$2"
  local batch_size="$3"
  local lr="$4"
  shift 4
  local extra_args=("$@")
  local log_file="$SUITE_DIR/${run_name}.log"

  echo ""
  echo "=================================================================="
  echo "RUN: $run_name"
  echo "LOG: $log_file"
  echo "=================================================================="

  set +e
  python experiments/train.py \
    "${COMMON_ARGS[@]}" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --run_name "$run_name" \
    "${extra_args[@]}" 2>&1 | tee "$log_file"
  local exit_code=${PIPESTATUS[0]}
  set -e

  if [[ $exit_code -eq 0 ]]; then
    append_summary "$run_name" "ok" "$log_file"
  else
    append_summary "$run_name" "fail($exit_code)" "$log_file"
  fi
}

if [[ "$MODE" == "tune" || "$MODE" == "all" ]]; then
  # Group A: split strategy
  run_one "tune_split_bearing_h${HORIZON}_bs${BATCH_SIZE}_lr${LR}" \
    "$EPOCHS_TUNE" "$BATCH_SIZE" "$LR" \
    --sound_split_mode bearing
  run_one "tune_split_loco_h${HORIZON}_bs${BATCH_SIZE}_lr${LR}" \
    "$EPOCHS_TUNE" "$BATCH_SIZE" "$LR" \
    --sound_split_mode leave_one_condition_out

  # Group B: batch/lr
  run_one "tune_bs64_lr1e3_h${HORIZON}" \
    "$EPOCHS_TUNE" "64" "0.001"
  run_one "tune_bs128_lr1e3_h${HORIZON}" \
    "$EPOCHS_TUNE" "128" "0.001"
  run_one "tune_bs128_lr1p5e3_h${HORIZON}" \
    "$EPOCHS_TUNE" "128" "0.0015"

  # Group C: horizon
  run_one "tune_h5_bs${BATCH_SIZE}_lr${LR}" \
    "$EPOCHS_TUNE" "$BATCH_SIZE" "$LR" \
    --horizon 5
  run_one "tune_h10_bs${BATCH_SIZE}_lr${LR}" \
    "$EPOCHS_TUNE" "$BATCH_SIZE" "$LR" \
    --horizon 10
  run_one "tune_h20_bs${BATCH_SIZE}_lr${LR}" \
    "$EPOCHS_TUNE" "$BATCH_SIZE" "$LR" \
    --horizon 20
fi

if [[ "$MODE" == "formal" || "$MODE" == "all" ]]; then
  # Formal runs: final main + bearing baseline
  run_one "formal_main_loco_h${HORIZON}_bs${BATCH_SIZE}_lr${LR}" \
    "$EPOCHS_FORMAL" "$BATCH_SIZE" "$LR" \
    --sound_split_mode leave_one_condition_out
  run_one "formal_baseline_bearing_h${HORIZON}_bs${BATCH_SIZE}_lr${LR}" \
    "$EPOCHS_FORMAL" "$BATCH_SIZE" "$LR" \
    --sound_split_mode bearing
fi

echo ""
echo "Suite finished."
echo "Summary: $SUMMARY_CSV"
echo "Top rows:"
if command -v column >/dev/null 2>&1; then
  column -s, -t "$SUMMARY_CSV" | sed -n '1,20p'
else
  sed -n '1,20p' "$SUMMARY_CSV"
fi
