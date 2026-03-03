# intraday_momentum_ml

Minimal scaffold for intraday momentum research with Alpaca 1-minute SPY data, feature engineering, baseline strategy tests, and ML filtering.

## Structure

- `src/`: Core Python modules.
- `.env.example`: Environment variable template.
- `requirements.txt`: Python dependencies.
- `.vscode/settings.json`: VS Code Python defaults.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set your real `ALPACA_API_KEY` and `ALPACA_API_SECRET`.
`ALPACA_DATA_URL` should usually stay `https://data.alpaca.markets`.

## CLI Entry Point

Run commands from the project root:

```bash
python -m src.cli <command> [options]
```

Example pipeline:

```bash
python -m src.cli fetch --symbol SPY --start 2025-01-01 --end 2025-03-01 --force
python -m src.cli preprocess
python -m src.cli baseline_backtest
python -m src.cli build_ml_dataset
python -m src.cli train_ml
python -m src.cli backtest_ml
```

Outputs are written under `DATA_DIR` (default `./data`) as parquet/json artifacts.

## Notebook Workflow

- Open [01_working_research.ipynb](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/notebooks/01_working_research.ipynb) in VS Code.
- In the notebook toolbar, click kernel picker and select `intraday-momentum-ml`.
- Run cells in order: Fetch -> Preprocess -> Indicators -> Baseline -> ML dataset -> Train -> ML backtest.
