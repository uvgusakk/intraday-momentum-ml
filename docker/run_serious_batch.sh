#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

mode="${BATCH_MODE:-full_pipeline}"
symbol="${BATCH_SYMBOL:-SPY}"
start="${BATCH_START:-2021-01-01}"
end="${BATCH_END:-2026-01-01}"
methods="${BATCH_METHODS:-soft,mm}"
label_mode="${BATCH_LABEL_MODE:-baseline_trade}"
horizon_mins="${BATCH_HORIZON_MINS:-60}"
train_months="${BATCH_TRAIN_MONTHS:-24}"
val_months="${BATCH_VAL_MONTHS:-3}"
test_months="${BATCH_TEST_MONTHS:-3}"
step_months="${BATCH_STEP_MONTHS:-3}"
warmup_days="${BATCH_WARMUP_DAYS:-90}"

fetch_args=(
  python -m src.cli fetch
  --symbol "$symbol"
  --start "$start"
  --end "$end"
)
if [[ "${BATCH_FETCH_FORCE:-0}" == "1" ]]; then
  fetch_args+=(--force)
fi

build_dataset_args=(
  python -m src.cli build_ml_dataset
  --label-mode "$label_mode"
  --horizon-mins "$horizon_mins"
)

train_args=(
  python -m src.cli train_ml
  --train-months "$train_months"
  --val-months "$val_months"
  --test-months "$test_months"
  --step-months "$step_months"
)

scoreforward_args=(
  python -m src.cli backtest_ml_scoreforward
  --methods "$methods"
  --label-mode "$label_mode"
  --horizon-mins "$horizon_mins"
  --train-months "$train_months"
  --val-months "$val_months"
  --test-months "$test_months"
  --step-months "$step_months"
  --warmup-days "$warmup_days"
)

echo "[batch] mode=$mode symbol=$symbol start=$start end=$end methods=$methods"
echo "[batch] label_mode=$label_mode horizon_mins=$horizon_mins train/val/test/step=${train_months}/${val_months}/${test_months}/${step_months}"

case "$mode" in
  full_pipeline)
    "${fetch_args[@]}"
    python -m src.cli preprocess
    "${build_dataset_args[@]}"
    "${train_args[@]}"
    "${scoreforward_args[@]}"
    ;;
  scoreforward_only)
    "${scoreforward_args[@]}"
    ;;
  *)
    echo "Unsupported BATCH_MODE: $mode" >&2
    echo "Allowed values: full_pipeline, scoreforward_only" >&2
    exit 2
    ;;
esac

python - <<'PY'
from pathlib import Path
import pandas as pd

summary_path = Path("data/ml_scoreforward_summary.csv")
if summary_path.exists():
    df = pd.read_csv(summary_path)
    cols = [c for c in ["method", "final_equity", "total_return_pct", "sharpe", "max_drawdown"] if c in df.columns]
    print("\n[batch] score-forward summary")
    print(df[cols].to_string(index=False))
else:
    print("\n[batch] summary file not found:", summary_path)
PY
