# intraday_momentum_ml

Single-symbol intraday momentum research for SPY with:

- a paper-style baseline strategy,
- an ML soft-sizing overlay,
- and one optional paper-inspired benchmark: market-volatility-managed sizing.

The current recommended strategy is:

1. `soft`
   - baseline direction,
   - 30-minute execution cadence,
   - execution-aligned `baseline_trade` labels,
   - aggressive soft-sizing map with `size_floor=0.15` and `size_cap=2.25`.

The current serious evaluation surface is intentionally narrow:

1. `baseline`
2. `soft` = the validated winning ML sizing strategy above
3. `mm` = `soft` plus a market-vol benchmark overlay

Other historical experiments remain in the repository for context, but they are
not treated as serious active methods.

Generated artifacts are written under `data/` and `reports/`. Those directories
are outputs of the research and reporting pipeline, not source-of-truth code.

Additional documentation:

- [System Overview](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/SYSTEM_OVERVIEW.md)
- [Cloud Run Proof](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/CLOUD_RUN_PROOF.md)

## Strategy Idea

### Baseline
The baseline follows the intraday Noise Area breakout structure from the SPY
intraday momentum paper:

- compute a 14-day minute-of-day sigma profile,
- build gap-adjusted upper/lower breakout bands (`UB`, `LB`),
- evaluate at semi-hourly decision times,
- enter in the breakout direction,
- stop using the tighter of the current band and VWAP,
- flatten by the close,
- include realistic trading costs.

### ML Soft Sizing
The ML layer does **not** replace the baseline signal. It only changes the size
of trades that the baseline already wants to take.

That design matters:

- the baseline still decides *direction*,
- the model only estimates *quality*,
- capital is scaled up or down smoothly instead of hard-filtering trades.

The best validated version now uses:

- the same 30-minute trading cadence as the baseline,
- a `baseline_trade` label built from the baseline's own realized trade outcome,
- `size_floor = 0.15`,
- `size_cap = 2.25`.

Under the realistic execution model, this was the strongest corrected
out-of-sample result found in the bounded strategy search.

### Market-Vol Overlay (`mm`)
The retained benchmark extension is a Moreira-Muir style market-volatility
overlay layered on top of soft sizing.

Idea:

- if recent broad market volatility is high, reduce exposure;
- if recent market volatility is calm, allow more exposure.

It still beats baseline, but it no longer beats the new best `soft`
configuration.

## Code Structure

The serious path is organized around a small set of modules:

- [src/baseline_strategy.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/baseline_strategy.py)
  - baseline wrapper and baseline-specific helpers
- [src/backtest_ml_filter.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/backtest_ml_filter.py)
  - ML sizing backtest on top of the baseline signal
- [src/scoreforward_eval.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/scoreforward_eval.py)
  - class-based rolling retrain / score-forward evaluation
- [src/features_ml.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/features_ml.py)
  - ML dataset construction
- [src/train_ml.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/train_ml.py)
  - walk-forward model training
- [src/ml_overlay_robust.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/ml_overlay_robust.py)
  - reusable exposure overlay helpers, including the market-vol overlay
- [src/cli.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/cli.py)
  - command-line entry points

The `ScoreforwardRunner` class in
[src/scoreforward_eval.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/scoreforward_eval.py)
keeps the evaluation flow explicit:

1. build the candidate ML dataset,
2. create chronological splits,
3. retrain on each split,
4. score only the next held-out window,
5. chain AUM forward,
6. summarize baseline vs `soft` vs `mm`.

## Environment Setup

From the project root:

```bash
cd /Users/ulianahusak/WUTIS_2026/intraday_momentum_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set:

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`

You usually do not need to change:

- `ALPACA_DATA_URL=https://data.alpaca.markets`
- `TZ=America/New_York`

## Step-by-Step Run Guide

### 1. Fetch data

```bash
python -m src.cli fetch --symbol SPY --start 2021-01-01 --end 2026-01-01 --force
```

Output:

- `data/bars_raw.parquet`

### 2. Preprocess and enrich

```bash
python -m src.cli preprocess
```

Outputs:

- `data/bars_preprocessed.parquet`
- `data/bars_enriched.parquet`

### 3. Run the paper baseline

```bash
python -m src.cli baseline_backtest
```

Outputs:

- `data/baseline_equity_curve.parquet`
- `data/baseline_trades.parquet`
- `data/baseline_summary.json`

### 4. Build the ML dataset

```bash
python -m src.cli build_ml_dataset
```

For the realistic score-forward selection path, the recommended label is
`baseline_trade`, not a fixed forward horizon.

The CLI default now follows that realistic choice.

Outputs:

- `data/ml_dataset.parquet`
- `data/ml_dataset_meta.json`

### 5. Train walk-forward ML models

```bash
python -m src.cli train_ml
```

Artifacts are written under:

- `data/models/`

### 6. Run the single best ML overlay (`soft`)

```bash
python -m src.cli backtest_ml
```

Default behavior now keeps the validated winning soft-sizing map:

- `size_floor = 0.15`
- `size_cap = 2.25`

Outputs:

- `data/ml_equity_curve.parquet`
- `data/ml_trades.parquet`
- `data/ml_metrics.json`
- `data/ml_vs_baseline.json`

### 7. Run the retained benchmark extension (`mm`)

```bash
python -m src.cli backtest_ml --market-vol-overlay
```

This uses the same baseline signal and the same ML soft-sizing path, with the
market-vol exposure overlay applied on top.

### 8. Run the corrected serious evaluation

This is the result you should trust for method selection:

```bash
python -m src.cli backtest_ml_scoreforward
```

This now defaults to the winning realistic `soft` strategy only. To compare the
market-vol benchmark as well:

```bash
python -m src.cli backtest_ml_scoreforward --methods soft,mm
```

Outputs:

- `data/ml_scoreforward_summary.csv`
- `data/ml_scoreforward_splits.csv`
- `data/ml_scoreforward/`

## How To Inspect Results In Terminal

Print the corrected summary:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/ml_scoreforward_summary.csv')
print(df[['method', 'final_equity', 'sharpe', 'max_drawdown']].to_string(index=False))
PY
```

Print the split-by-split view:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/ml_scoreforward_splits.csv')
print(df[['split_id', 'method', 'final_equity', 'sharpe', 'max_drawdown']].to_string(index=False))
PY
```

## Docker Batch Run

The repository includes a narrow container entrypoint for the current serious
batch path only. It does not add any live-trading behavior; it just runs the
existing CLI pipeline inside Docker.

Files:

- [Dockerfile](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/Dockerfile)
- [run_serious_batch.sh](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docker/run_serious_batch.sh)
- [.dockerignore](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/.dockerignore)

Build the image:

```bash
cd /Users/ulianahusak/WUTIS_2026/intraday_momentum_ml
docker build -t intraday-momentum-ml:local .
```

Run a quick local smoke batch without touching your local `data/` metrics:

```bash
mkdir -p /tmp/intraday-momentum-ml-docker-data

docker run --rm \
  --env-file .env \
  -e DATA_DIR=/app/data \
  -e BATCH_MODE=full_pipeline \
  -e BATCH_SYMBOL=SPY \
  -e BATCH_START=2025-01-01 \
  -e BATCH_END=2026-01-01 \
  -e BATCH_TRAIN_MONTHS=6 \
  -e BATCH_VAL_MONTHS=1 \
  -e BATCH_TEST_MONTHS=1 \
  -e BATCH_STEP_MONTHS=1 \
  -e BATCH_METHODS=soft \
  -v /tmp/intraday-momentum-ml-docker-data:/app/data \
  intraday-momentum-ml:local
```

That entrypoint defaults to `full_pipeline`, which runs:

1. `fetch`
2. `preprocess`
3. `build_ml_dataset`
4. `train_ml`
5. `backtest_ml_scoreforward`

If you already have prepared data and just want the final batch run, use:

```bash
docker run --rm \
  --env-file .env \
  -e DATA_DIR=/app/data \
  -e BATCH_MODE=scoreforward_only \
  -e BATCH_METHODS=soft,mm \
  -v /absolute/path/to/data:/app/data \
  intraday-momentum-ml:local
```

To inspect the container without running the batch entrypoint, override the
command:

```bash
docker run --rm intraday-momentum-ml:local python -m src.cli --help
```

## Live Websocket Runtime

The research and score-forward metrics remain unchanged. Live and paper-routing
behavior now lives in a separate websocket path:

- [src/live_alpaca.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/live_alpaca.py)
  - Alpaca historical warmup, websocket bars, and paper broker adapter
- [src/live_strategy_runtime.py](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/src/live_strategy_runtime.py)
  - baseline plus the top 3 realistic live variants:
    - `baseline`
    - `soft_hybrid_7_5`
    - `soft_hybrid_10`
    - `soft_hybrid_5`

### Live strategy board

Monitor all four variants together over the Alpaca websocket and save the
latest board under `data/live/`:

```bash
python -m src.cli live_strategy_board \
  --symbol SPY \
  --variants baseline,soft_hybrid_7_5,soft_hybrid_10,soft_hybrid_5 \
  --with-account \
  --duration-seconds 300
```

Outputs:

- `data/live/live_strategy_board_latest.csv`
- `data/live/live_strategy_board_latest.json`
- `data/live/live_strategy_board_history.parquet`

### Live paper run for one selected strategy

Route one strategy at a time to Alpaca paper trading:

```bash
python -m src.cli live_paper_strategy \
  --variant soft_hybrid_7_5 \
  --symbol SPY \
  --duration-seconds 300
```

For a dry run with no paper orders:

```bash
python -m src.cli live_paper_strategy \
  --variant soft_hybrid_7_5 \
  --symbol SPY \
  --duration-seconds 300 \
  --dry-run
```

Outputs:

- `data/live/soft_hybrid_7_5_live_runtime.parquet`
- `data/live/soft_hybrid_7_5_live_runtime_latest.json`

Docker helper for the live board:

```bash
docker run --rm \
  --env-file .env \
  -e DATA_DIR=/app/data \
  -e LIVE_DURATION_SECONDS=300 \
  -v /tmp/intraday-momentum-ml-docker-data:/app/data \
  intraday-momentum-ml:local \
  /usr/local/bin/run-live-strategy
```

## GCP Deployment Path

The narrow deployment target for the current serious path is:

1. Artifact Registry for the image
2. Cloud Run Jobs for batch execution
3. Secret Manager for Alpaca credentials

This keeps deployment aligned with the existing batch-oriented workflow.

For the live websocket runtime, keep the code separate from the batch job. A
live deployment should use a dedicated long-lived runtime target after the
paper-routing path is validated locally.

Helper scripts:

- [create_secrets_from_env.sh](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/gcp/create_secrets_from_env.sh)
- [deploy_cloud_run_job.sh](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/gcp/deploy_cloud_run_job.sh)

### 1. Authenticate gcloud

```bash
gcloud auth login
gcloud auth application-default login
```

### 2. Set your project and region

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="europe-west1"
```

### 3. Create Alpaca secrets from your local `.env`

```bash
./gcp/create_secrets_from_env.sh
```

This uploads:

- `alpaca-api-key`
- `alpaca-api-secret`

### 4. Deploy or update the Cloud Run Job

```bash
chmod +x gcp/*.sh

export BATCH_MODE="scoreforward_only"
export BATCH_METHODS="soft,mm"

./gcp/deploy_cloud_run_job.sh
```

The script:

1. enables required Google APIs
2. creates Artifact Registry if needed
3. builds the Docker image locally
4. pushes it to Artifact Registry
5. creates or updates a Cloud Run Job

By default the Cloud Run Job uses:

- `DATA_DIR=/tmp/data`
- `BATCH_MODE=scoreforward_only`
- `BATCH_METHODS=soft,mm`

The container writes job outputs to ephemeral storage and logs the final summary
to Cloud Logging. If you later want durable artifacts, the next step is to add
Cloud Storage export.

### 5. Execute the Cloud Run Job

```bash
gcloud run jobs execute intraday-momentum-ml-batch --region "$GCP_REGION" --wait
```

### 6. Inspect logs

```bash
gcloud run jobs executions list --job intraday-momentum-ml-batch --region "$GCP_REGION"
```

You can also inspect the job and execution logs in the Google Cloud Console.

## Expected Serious-Method Ordering

With the corrected rolling retrain / score-forward evaluation, the validated
ordering is:

1. `soft`
2. `baseline`
3. `mm`

At the time of the last checked realistic run, the summary was:

| method   | final_equity | sharpe   | max_drawdown |
|----------|-------------:|---------:|-------------:|
| soft     | 120396.9044  |  0.383525 | -0.202688    |
| mm       |  98606.6743  |  0.037862 | -0.162954    |
| baseline |  91686.8938  | -0.171875 | -0.187587    |

If your numbers diverge materially, check:

- that you are using the same date range,
- that the enriched bars were rebuilt from the same raw cache,
- and that the walk-forward model artifacts were retrained before the backtest.

## Notebooks

The active notebook set is:

- [notebooks/01_working_research.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/01_working_research.ipynb)
  - end-to-end research workflow and the Alpaca live notebook demo section
- [notebooks/03_realistic_soft_improvements.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/03_realistic_soft_improvements.ipynb)
  - realistic `soft` / hybrid-stop comparison and path diagnostics
- [notebooks/04_unseen_2025_holdout_check.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/04_unseen_2025_holdout_check.ipynb)
  - whole-timeline and holdout robustness checks
