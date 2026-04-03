#!/usr/bin/env bash
set -euo pipefail

cd /app

exec python -m src.cli live_strategy_board \
  --symbol "${LIVE_SYMBOL:-SPY}" \
  --variants "${LIVE_VARIANTS:-baseline,soft_hybrid_7_5,soft_hybrid_10,soft_hybrid_5}" \
  --history-business-days "${LIVE_HISTORY_BUSINESS_DAYS:-20}" \
  --refresh-seconds "${LIVE_REFRESH_SECONDS:-5}" \
  --duration-seconds "${LIVE_DURATION_SECONDS:-300}" \
  "${@}"
