# System Overview

This repository now has three clearly separated layers:

1. Research and backtesting
2. Live websocket and Alpaca paper-routing
3. Container and Google Cloud deployment

The separation is intentional. The live/runtime work does not overwrite the
validated research metrics path.

## 1. Research and Backtesting

Canonical research modules:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/baseline_strategy.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/baseline_strategy.py)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/backtest_ml_filter.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/backtest_ml_filter.py)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/scoreforward_eval.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/scoreforward_eval.py)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/features_ml.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/features_ml.py)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/train_ml.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/train_ml.py)

Canonical current realistic score-forward summary:

| method | total_return_pct | sharpe | max_drawdown |
|---|---:|---:|---:|
| `soft` | `20.396904` | `0.383525` | `-0.202688` |
| `mm` | `-1.393326` | `0.037862` | `-0.162954` |
| `baseline` | `-8.313106` | `-0.171875` | `-0.187587` |

Those numbers come from:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/data/ml_scoreforward_summary.csv](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/data/ml_scoreforward_summary.csv)

## 2. Live Websocket and Paper Routing

The live path is separate from the research path.

Core live modules:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/live_alpaca.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/live_alpaca.py)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/live_strategy_runtime.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/live_strategy_runtime.py)

Available live variants:

- `baseline`
- `soft_hybrid_7_5`
- `soft_hybrid_10`
- `soft_hybrid_5`

CLI entrypoints:

- `python -m src.cli live_strategy_board`
- `python -m src.cli live_paper_strategy`

What this layer does:

- seeds historical warmup bars from Alpaca
- connects to Alpaca websocket minute bars
- computes the current strategy state on fresh live bars
- optionally routes a selected strategy to Alpaca paper trading

What this layer does not do:

- it does not rewrite backtest outputs
- it does not redefine the score-forward metrics

## 3. Container and Cloud Deployment

Container files:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/Dockerfile](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/Dockerfile)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/.dockerignore](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/.dockerignore)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docker/run_serious_batch.sh](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docker/run_serious_batch.sh)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docker/run_live_strategy.sh](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docker/run_live_strategy.sh)

Google Cloud helper files:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/gcp/create_secrets_from_env.sh](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/gcp/create_secrets_from_env.sh)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/gcp/deploy_cloud_run_job.sh](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/gcp/deploy_cloud_run_job.sh)

Current deployment status:

- batch image built and pushed to Artifact Registry
- Cloud Run Job deployed and executed successfully
- raw proof artifacts saved under `docs/cloud_run_proof/`

See:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/CLOUD_RUN_PROOF.md](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/CLOUD_RUN_PROOF.md)

## Notebooks

Active notebook set:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/01_working_research.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/01_working_research.ipynb)
  - end-to-end research workflow plus a final Alpaca live demo section
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/03_realistic_soft_improvements.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/03_realistic_soft_improvements.ipynb)
  - realistic hybrid-stop comparison and path diagnostics
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/04_unseen_2025_holdout_check.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/04_unseen_2025_holdout_check.ipynb)
  - whole-timeline and holdout robustness checks

## Output Directories

- `data/`
  - cached historical data, enriched bars, model artifacts, score-forward outputs
- `data/live/`
  - live board snapshots and live runtime outputs
- `reports/`
  - notebook and search outputs
- `docs/cloud_run_proof/`
  - deployment evidence captured from Google Cloud
