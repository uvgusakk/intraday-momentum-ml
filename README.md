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

## Notebook

The current research notebook is:

- [notebooks/02_strategy_selection_and_update_report.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/02_strategy_selection_and_update_report.ipynb)

It documents:

- why the evaluation protocol was corrected,
- what the paper-inspired overlays were,
- why realistic execution materially reduced the optimistic backtest metrics,
- why the execution-aligned `baseline_trade` label replaced the earlier
  fixed-horizon label in the serious path,
- why `soft` is still the recommended realistic strategy,
- and how the final ranking was established.
