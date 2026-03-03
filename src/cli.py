"""Command-line entry points for intraday momentum research workflows."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest_ml_filter import run_ml_filtered_backtest
from .baseline_strategy import run_baseline_backtest
from .config import load_config
from .data_alpaca import fetch_minute_bars
from .features_ml import build_ml_dataset
from .indicators import (
    compute_gap_adjusted_bands,
    compute_intraday_move_from_open,
    compute_sigma_profile,
    compute_vwap,
)
from .preprocess import preprocess_bars
from .train_ml import train_walk_forward_models

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _paths() -> dict[str, Path]:
    config = load_config()
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "raw": data_dir / "bars_raw.parquet",
        "preprocessed": data_dir / "bars_preprocessed.parquet",
        "enriched": data_dir / "bars_enriched.parquet",
        "baseline_equity": data_dir / "baseline_equity_curve.parquet",
        "baseline_trades": data_dir / "baseline_trades.parquet",
        "baseline_summary": data_dir / "baseline_summary.json",
        "ml_dataset": data_dir / "ml_dataset.parquet",
        "ml_dataset_meta": data_dir / "ml_dataset_meta.json",
        "ml_equity": data_dir / "ml_equity_curve.parquet",
        "ml_trades": data_dir / "ml_trades.parquet",
        "ml_metrics": data_dir / "ml_metrics.json",
        "ml_comparison": data_dir / "ml_vs_baseline.parquet",
        "ml_comparison_json": data_dir / "ml_vs_baseline.json",
        "ml_decisions": data_dir / "ml_candidate_decisions.parquet",
    }


def _read_parquet_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _enrich_bars(preprocessed: pd.DataFrame) -> pd.DataFrame:
    enriched = compute_intraday_move_from_open(preprocessed)
    enriched = compute_sigma_profile(enriched, lookback_days=14)
    enriched = compute_gap_adjusted_bands(enriched, vm=1.0)
    enriched = compute_vwap(enriched)
    return enriched


def cmd_fetch(args: argparse.Namespace) -> None:
    p = _paths()
    bars = fetch_minute_bars(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        adjustment=args.adjustment,
        force=args.force,
    )
    bars.to_parquet(p["raw"], index=False)

    _dump_json(
        p["data_dir"] / "fetch_meta.json",
        {
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "adjustment": args.adjustment,
            "force": args.force,
            "rows": int(len(bars)),
            "output": str(p["raw"]),
        },
    )
    logger.info("Saved raw bars to %s (%d rows)", p["raw"], len(bars))


def cmd_preprocess(_: argparse.Namespace) -> None:
    p = _paths()
    raw = _read_parquet_required(p["raw"])
    clean = preprocess_bars(raw)
    clean.to_parquet(p["preprocessed"], index=False)

    enriched = _enrich_bars(clean)
    enriched.to_parquet(p["enriched"], index=False)

    _dump_json(
        p["data_dir"] / "preprocess_meta.json",
        {
            "rows_raw": int(len(raw)),
            "rows_preprocessed": int(len(clean)),
            "rows_enriched": int(len(enriched)),
            "preprocessed_output": str(p["preprocessed"]),
            "enriched_output": str(p["enriched"]),
        },
    )
    logger.info("Saved preprocessed bars to %s", p["preprocessed"])
    logger.info("Saved enriched bars to %s", p["enriched"])


def cmd_baseline_backtest(args: argparse.Namespace) -> None:
    p = _paths()
    bars = _read_parquet_required(p["enriched"])

    out = run_baseline_backtest(
        bars,
        initial_aum=args.initial_aum,
        sigma_target=args.sigma_target,
        lev_cap=args.lev_cap,
        commission_per_share=args.commission_per_share,
        slippage_per_share=args.slippage_per_share,
        decision_freq_mins=args.decision_freq_mins,
        first_trade_time=args.first_trade_time,
    )

    out["equity_curve"].to_parquet(p["baseline_equity"], index=False)
    out["trades"].to_parquet(p["baseline_trades"], index=False)
    _dump_json(p["baseline_summary"], out["summary"])

    logger.info("Saved baseline equity to %s", p["baseline_equity"])
    logger.info("Saved baseline trades to %s", p["baseline_trades"])
    logger.info("Saved baseline summary to %s", p["baseline_summary"])


def cmd_build_ml_dataset(args: argparse.Namespace) -> None:
    p = _paths()
    bars = _read_parquet_required(p["enriched"])

    X, y, meta = build_ml_dataset(
        bars,
        backtest_kwargs={
            "initial_aum": args.initial_aum,
            "sigma_target": args.sigma_target,
            "lev_cap": args.lev_cap,
            "commission_per_share": args.commission_per_share,
            "slippage_per_share": args.slippage_per_share,
            "decision_freq_mins": args.decision_freq_mins,
            "first_trade_time": args.first_trade_time,
        },
        label_mode=args.label_mode,
        horizon_mins=args.horizon_mins,
    )

    _dump_json(
        p["ml_dataset_meta"],
        {
            "rows": int(len(X)),
            "features": list(X.columns),
            "positive_rate": float(y.mean()) if len(y) else 0.0,
            "dataset_path": str(p["ml_dataset"]),
            "meta_columns": list(meta.columns),
        },
    )
    logger.info("ML dataset saved to %s", p["ml_dataset"])
    logger.info("ML dataset metadata saved to %s", p["ml_dataset_meta"])


def cmd_train_ml(_: argparse.Namespace) -> None:
    out = train_walk_forward_models(
        train_months=_.train_months,
        val_months=_.val_months,
        test_months=_.test_months,
        step_months=_.step_months,
    )
    logger.info("Training complete. Report: %s", out["report_path"])


def cmd_backtest_ml(args: argparse.Namespace) -> None:
    p = _paths()
    bars = _read_parquet_required(p["enriched"])

    out = run_ml_filtered_backtest(
        bars,
        initial_aum=args.initial_aum,
        sigma_target=args.sigma_target,
        lev_cap=args.lev_cap,
        commission_per_share=args.commission_per_share,
        slippage_per_share=args.slippage_per_share,
        decision_freq_mins=args.decision_freq_mins,
        first_trade_time=args.first_trade_time,
        model_path=args.model_path,
        calibration_path=args.calibration_path,
        threshold_path=args.threshold_path,
        flip_reject_mode=args.flip_reject_mode,
        filter_mode="entry_only",
        allocation_mode="soft_size",
        size_floor=args.size_floor,
        size_cap=args.size_cap,
        neutral_zone=not args.no_neutral_zone,
        regime_overlay=not args.no_regime_overlay,
        regime_lookback_months=args.regime_lookback_months,
        regime_min_trades=args.regime_min_trades,
    )

    out["equity_curve"].to_parquet(p["ml_equity"], index=False)
    out["trades"].to_parquet(p["ml_trades"], index=False)
    out["comparison"].to_parquet(p["ml_comparison"], index=True)

    if not out["candidate_decisions"].empty:
        out["candidate_decisions"].to_parquet(p["ml_decisions"], index=False)

    _dump_json(p["ml_metrics"], out["metrics"])
    _dump_json(
        p["ml_comparison_json"],
        {
            "baseline_summary": out["baseline_summary"],
            "ml_metrics": out["metrics"],
            "comparison": out["comparison"].to_dict(orient="index"),
        },
    )

    logger.info("Saved ML equity to %s", p["ml_equity"])
    logger.info("Saved ML trades to %s", p["ml_trades"])
    logger.info("Saved ML metrics to %s", p["ml_metrics"])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="intraday_momentum_ml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch 1-minute bars from Alpaca")
    p_fetch.add_argument("--symbol", required=True)
    p_fetch.add_argument("--start", required=True, help="YYYY-MM-DD or timestamp")
    p_fetch.add_argument("--end", required=True, help="YYYY-MM-DD or timestamp")
    p_fetch.add_argument("--adjustment", default="raw")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_fetch)

    p_pre = sub.add_parser("preprocess", help="Preprocess raw bars and compute indicators")
    p_pre.set_defaults(func=cmd_preprocess)

    p_base = sub.add_parser("baseline_backtest", help="Run baseline backtest")
    p_base.add_argument("--initial-aum", type=float, default=100000)
    p_base.add_argument("--sigma-target", type=float, default=0.02)
    p_base.add_argument("--lev-cap", type=float, default=4.0)
    p_base.add_argument("--commission-per-share", type=float, default=0.0035)
    p_base.add_argument("--slippage-per-share", type=float, default=0.001)
    p_base.add_argument("--decision-freq-mins", type=int, default=30)
    p_base.add_argument("--first-trade-time", default="10:00")
    p_base.set_defaults(func=cmd_baseline_backtest)

    p_ds = sub.add_parser("build_ml_dataset", help="Build ML dataset from baseline entries")
    p_ds.add_argument("--initial-aum", type=float, default=100000)
    p_ds.add_argument("--sigma-target", type=float, default=0.02)
    p_ds.add_argument("--lev-cap", type=float, default=4.0)
    p_ds.add_argument("--commission-per-share", type=float, default=0.0035)
    p_ds.add_argument("--slippage-per-share", type=float, default=0.001)
    p_ds.add_argument("--decision-freq-mins", type=int, default=30)
    p_ds.add_argument("--first-trade-time", default="10:00")
    p_ds.add_argument("--label-mode", choices=["baseline_trade", "fixed_horizon"], default="fixed_horizon")
    p_ds.add_argument("--horizon-mins", type=int, default=30)
    p_ds.set_defaults(func=cmd_build_ml_dataset)

    p_train = sub.add_parser("train_ml", help="Train walk-forward ML models")
    p_train.add_argument("--train-months", type=int, default=24)
    p_train.add_argument("--val-months", type=int, default=3)
    p_train.add_argument("--test-months", type=int, default=3)
    p_train.add_argument("--step-months", type=int, default=3)
    p_train.set_defaults(func=cmd_train_ml)

    p_bt_ml = sub.add_parser("backtest_ml", help="Run ML-filtered backtest")
    p_bt_ml.add_argument("--initial-aum", type=float, default=100000)
    p_bt_ml.add_argument("--sigma-target", type=float, default=0.02)
    p_bt_ml.add_argument("--lev-cap", type=float, default=4.0)
    p_bt_ml.add_argument("--commission-per-share", type=float, default=0.0035)
    p_bt_ml.add_argument("--slippage-per-share", type=float, default=0.001)
    p_bt_ml.add_argument("--decision-freq-mins", type=int, default=30)
    p_bt_ml.add_argument("--first-trade-time", default="10:00")
    p_bt_ml.add_argument("--model-path", default=None)
    p_bt_ml.add_argument("--calibration-path", default=None)
    p_bt_ml.add_argument("--threshold-path", default=None)
    p_bt_ml.add_argument("--flip-reject-mode", choices=["hold", "close_flat"], default="hold")
    # CLI path is soft-sizing only; hard-filter remains notebook diagnostics.
    p_bt_ml.add_argument("--size-floor", type=float, default=0.85)
    p_bt_ml.add_argument("--size-cap", type=float, default=1.15)
    p_bt_ml.add_argument("--no-neutral-zone", action="store_true")
    p_bt_ml.add_argument("--no-regime-overlay", action="store_true")
    p_bt_ml.add_argument("--regime-lookback-months", type=int, default=6)
    p_bt_ml.add_argument("--regime-min-trades", type=int, default=80)
    p_bt_ml.set_defaults(func=cmd_backtest_ml)

    return parser


def main() -> None:
    """Run CLI command dispatch."""
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
