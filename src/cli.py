"""Command-line entry points for intraday momentum research workflows."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest_ml_filter import _build_feature_row, _load_artifacts, _raw_scores, run_ml_filtered_backtest
from .baseline_strategy import run_baseline_backtest
from .config import load_config
from .config import (
    DEFAULT_DECISION_FREQ_MINS,
    DEFAULT_EXECUTION_SPREAD_BPS,
    DEFAULT_ML_LABEL_HORIZON_MINS,
    DEFAULT_MINUTE_STOP_MONITORING,
    DEFAULT_MM_CAP,
    DEFAULT_MM_FLOOR,
    DEFAULT_MM_LOOKBACK_DAYS,
    DEFAULT_SCOREFORWARD_LABEL_MODE,
    DEFAULT_SOFT_SIZE_CAP,
    DEFAULT_SOFT_SIZE_FLOOR,
    DEFAULT_USE_NEXT_BAR_OPEN,
)
from .data_alpaca import fetch_minute_bars
from .engine.backtest_engine import BacktestConfig, BacktestEngine, FixedQuantityRiskManager
from .features_ml import build_ml_dataset
from .indicators import (
    compute_gap_adjusted_bands,
    compute_intraday_move_from_open,
    compute_sigma_profile,
    compute_vwap,
)
from .live_alpaca import AlpacaLiveMarketData, AlpacaPaperBroker
from .live_strategy_runtime import (
    LivePaperStrategyRunner,
    list_live_variants,
    run_live_strategy_board_loop,
)
from .preprocess import preprocess_bars
from .scoreforward_eval import ScoreforwardConfig, run_ml_scoreforward_backtests
from .strategies import BaselineNoiseAreaStrategy, MLOutputSizerRiskManager
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
        "ml_scoreforward_summary": data_dir / "ml_scoreforward_summary.csv",
        "ml_scoreforward_splits": data_dir / "ml_scoreforward_splits.csv",
        "ml_scoreforward_dir": data_dir / "ml_scoreforward",
        "live_dir": data_dir / "live",
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
    enriched = compute_gap_adjusted_bands(enriched)
    enriched = compute_vwap(enriched)
    return enriched


def _add_realism_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-next-bar-open",
        dest="use_next_bar_open",
        action="store_true",
        default=DEFAULT_USE_NEXT_BAR_OPEN,
        help="Fill decision-time entries and flips at the next minute open.",
    )
    parser.add_argument(
        "--no-next-bar-open",
        dest="use_next_bar_open",
        action="store_false",
        help="Use same-bar close fills instead of next-bar open execution.",
    )
    parser.add_argument(
        "--minute-stop-monitoring",
        dest="minute_stop_monitoring",
        action="store_true",
        default=DEFAULT_MINUTE_STOP_MONITORING,
        help="Check stops on every minute bar instead of only decision timestamps.",
    )
    parser.add_argument(
        "--no-minute-stop-monitoring",
        dest="minute_stop_monitoring",
        action="store_false",
        help="Only check stops at decision timestamps.",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=DEFAULT_EXECUTION_SPREAD_BPS,
        help="Full bid/ask spread proxy in basis points; half-spread is charged on each fill.",
    )


def _add_realistic_soft_improvement_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execution-chase-control",
        action="store_true",
        help="Downsize entries when the realized next-bar fill drifts too far against the signal close.",
    )
    parser.add_argument("--execution-chase-bps", type=float, default=8.0)
    parser.add_argument("--execution-chase-mult", type=float, default=0.5)
    parser.add_argument(
        "--execution-chase-relative",
        action="store_true",
        help="Use a band-width / intraday-vol relative entry drift threshold instead of a raw bps threshold.",
    )
    parser.add_argument("--execution-chase-band-frac", type=float, default=0.15)
    parser.add_argument("--execution-chase-vol-mult", type=float, default=0.75)
    parser.add_argument(
        "--hybrid-stop-mode",
        action="store_true",
        help="Only trigger minute-level stops for catastrophic stop breaches; keep standard stop checks at decision times.",
    )
    parser.add_argument("--catastrophic-stop-bps", type=float, default=10.0)
    parser.add_argument(
        "--intraday-risk-overlay",
        action="store_true",
        help="Downsize ML exposure when current intraday volatility is elevated versus recent candidate history.",
    )
    parser.add_argument("--intraday-risk-lookback-days", type=int, default=60)
    parser.add_argument("--intraday-risk-quantile", type=float, default=0.8)
    parser.add_argument("--intraday-risk-downsize-mult", type=float, default=0.75)



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

    preprocess_meta: dict[str, Any] = {
        "rows_raw": int(len(raw)),
        "rows_preprocessed": int(len(clean)),
        "rows_enriched": int(len(enriched)),
        "preprocessed_output": str(p["preprocessed"]),
        "enriched_output": str(p["enriched"]),
    }

    _dump_json(p["data_dir"] / "preprocess_meta.json", preprocess_meta)
    logger.info("Saved default enriched bars to %s", p["enriched"])
    logger.info("Saved preprocessed bars to %s", p["preprocessed"])


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
        use_next_bar_open=args.use_next_bar_open,
        minute_stop_monitoring=args.minute_stop_monitoring,
        spread_bps=args.spread_bps,
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
            "use_next_bar_open": args.use_next_bar_open,
            "minute_stop_monitoring": args.minute_stop_monitoring,
            "spread_bps": args.spread_bps,
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
        target_mode=_.target_mode,
        target_quantile=_.target_quantile,
        side_filter=_.side_filter,
        artifact_subdir=_.artifact_subdir,
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
        filter_mode="entry_only",
        allocation_mode="soft_size",
        size_floor=args.size_floor,
        size_cap=args.size_cap,
        neutral_zone=True,
        regime_overlay=False,
        market_vol_overlay=args.market_vol_overlay,
        market_vol_lookback_days=args.market_vol_lookback_days,
        market_vol_floor=args.market_vol_floor,
        market_vol_cap=args.market_vol_cap,
        use_next_bar_open=args.use_next_bar_open,
        minute_stop_monitoring=args.minute_stop_monitoring,
        spread_bps=args.spread_bps,
        execution_chase_control=args.execution_chase_control,
        execution_chase_bps=args.execution_chase_bps,
        execution_chase_mult=args.execution_chase_mult,
        execution_chase_relative=args.execution_chase_relative,
        execution_chase_band_frac=args.execution_chase_band_frac,
        execution_chase_vol_mult=args.execution_chase_vol_mult,
        hybrid_stop_mode=args.hybrid_stop_mode,
        catastrophic_stop_bps=args.catastrophic_stop_bps,
        intraday_risk_overlay=args.intraday_risk_overlay,
        intraday_risk_lookback_days=args.intraday_risk_lookback_days,
        intraday_risk_quantile=args.intraday_risk_quantile,
        intraday_risk_downsize_mult=args.intraday_risk_downsize_mult,
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


def cmd_backtest_ml_scoreforward(args: argparse.Namespace) -> None:
    p = _paths()
    bars = _read_parquet_required(p["enriched"])
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    overrides: dict[str, dict[str, Any]] = {
        "soft": {
            "execution_chase_control": args.execution_chase_control,
            "execution_chase_bps": args.execution_chase_bps,
            "execution_chase_mult": args.execution_chase_mult,
            "execution_chase_relative": args.execution_chase_relative,
            "execution_chase_band_frac": args.execution_chase_band_frac,
            "execution_chase_vol_mult": args.execution_chase_vol_mult,
            "hybrid_stop_mode": args.hybrid_stop_mode,
            "catastrophic_stop_bps": args.catastrophic_stop_bps,
            "intraday_risk_overlay": args.intraday_risk_overlay,
            "intraday_risk_lookback_days": args.intraday_risk_lookback_days,
            "intraday_risk_quantile": args.intraday_risk_quantile,
            "intraday_risk_downsize_mult": args.intraday_risk_downsize_mult,
        },
        "mm": {
            "market_vol_lookback_days": args.market_vol_lookback_days,
            "market_vol_floor": args.market_vol_floor,
            "market_vol_cap": args.market_vol_cap,
            "execution_chase_control": args.execution_chase_control,
            "execution_chase_bps": args.execution_chase_bps,
            "execution_chase_mult": args.execution_chase_mult,
            "execution_chase_relative": args.execution_chase_relative,
            "execution_chase_band_frac": args.execution_chase_band_frac,
            "execution_chase_vol_mult": args.execution_chase_vol_mult,
            "hybrid_stop_mode": args.hybrid_stop_mode,
            "catastrophic_stop_bps": args.catastrophic_stop_bps,
            "intraday_risk_overlay": args.intraday_risk_overlay,
            "intraday_risk_lookback_days": args.intraday_risk_lookback_days,
            "intraday_risk_quantile": args.intraday_risk_quantile,
            "intraday_risk_downsize_mult": args.intraday_risk_downsize_mult,
        },
    }
    out = run_ml_scoreforward_backtests(
        bars,
        config=ScoreforwardConfig(
            initial_aum=args.initial_aum,
            sigma_target=args.sigma_target,
            lev_cap=args.lev_cap,
            commission_per_share=args.commission_per_share,
            slippage_per_share=args.slippage_per_share,
            decision_freq_mins=args.decision_freq_mins,
            first_trade_time=args.first_trade_time,
            use_next_bar_open=args.use_next_bar_open,
            minute_stop_monitoring=args.minute_stop_monitoring,
            spread_bps=args.spread_bps,
            label_mode=args.label_mode,
            horizon_mins=args.horizon_mins,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            step_months=args.step_months,
            warmup_days=args.warmup_days,
        ),
        methods=methods,
        method_overrides=overrides,
    )

    p["ml_scoreforward_dir"].mkdir(parents=True, exist_ok=True)
    out["summary"].to_csv(p["ml_scoreforward_summary"], index=False)
    out["split_metrics"].to_csv(p["ml_scoreforward_splits"], index=False)
    for method, payload in out["outputs"].items():
        payload["equity_curve"].to_parquet(p["ml_scoreforward_dir"] / f"{method}_equity_curve.parquet", index=False)
        payload["trades"].to_parquet(p["ml_scoreforward_dir"] / f"{method}_trades.parquet", index=False)
        _dump_json(p["ml_scoreforward_dir"] / f"{method}_summary.json", payload["summary"])

    logger.info("Saved score-forward summary to %s", p["ml_scoreforward_summary"])
    logger.info("Saved score-forward split metrics to %s", p["ml_scoreforward_splits"])


def _parse_csv_variants(raw: str | None) -> list[str]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return values or list_live_variants()


def cmd_live_strategy_board(args: argparse.Namespace) -> None:
    p = _paths()
    live_dir = p["live_dir"]
    live_dir.mkdir(parents=True, exist_ok=True)

    market_data = AlpacaLiveMarketData(
        symbol=args.symbol,
        feed=args.feed,
        history_business_days=args.history_business_days,
    )
    seeded = market_data.seed_history(force=args.force_seed)
    seed_cutoff_ts = pd.Timestamp(seeded["timestamp"].max()) if not seeded.empty else None

    broker = AlpacaPaperBroker() if args.with_account else None
    try:
        rows = run_live_strategy_board_loop(
            market_data=market_data,
            broker=broker,
            strategy_names=_parse_csv_variants(args.variants),
            symbol=args.symbol,
            refresh_seconds=args.refresh_seconds,
            duration_seconds=args.duration_seconds,
            output_dir=live_dir,
            min_live_timestamp=seed_cutoff_ts,
        )
        if not rows.empty:
            rows.to_parquet(live_dir / "live_strategy_board_history.parquet", index=False)
            logger.info("Saved live strategy board history to %s", live_dir / "live_strategy_board_history.parquet")
    finally:
        market_data.stop()
        if broker is not None:
            broker.stop_trade_updates()


def cmd_live_paper_strategy(args: argparse.Namespace) -> None:
    p = _paths()
    live_dir = p["live_dir"]
    live_dir.mkdir(parents=True, exist_ok=True)

    market_data = AlpacaLiveMarketData(
        symbol=args.symbol,
        feed=args.feed,
        history_business_days=args.history_business_days,
    )
    seeded = market_data.seed_history(force=args.force_seed)
    seed_cutoff_ts = pd.Timestamp(seeded["timestamp"].max()) if not seeded.empty else None
    broker = AlpacaPaperBroker(allow_live=args.allow_live)
    runner = LivePaperStrategyRunner(
        market_data=market_data,
        broker=broker,
        variant_name=args.variant,
        symbol=args.symbol,
        sigma_target=args.sigma_target,
        lev_cap=args.lev_cap,
        dry_run=args.dry_run,
        min_live_timestamp=seed_cutoff_ts,
    )
    try:
        rows = runner.run(
            refresh_seconds=args.refresh_seconds,
            duration_seconds=args.duration_seconds,
        )
        if not rows.empty:
            rows.to_parquet(live_dir / f"{args.variant}_live_runtime.parquet", index=False)
            _dump_json(
                live_dir / f"{args.variant}_live_runtime_latest.json",
                rows.iloc[-1].to_dict(),
            )
            logger.info("Saved live runtime snapshots to %s", live_dir / f"{args.variant}_live_runtime.parquet")
    finally:
        market_data.stop()
        broker.stop_trade_updates()


def cmd_run_system_baseline_engine(args: argparse.Namespace) -> None:
    p = _paths()
    bars = _read_parquet_required(p["enriched"])

    engine = BacktestEngine(
        strategy=BaselineNoiseAreaStrategy(
            decision_freq_mins=args.decision_freq_mins,
            first_trade_time=args.first_trade_time,
        ),
        risk_manager=FixedQuantityRiskManager(),
        config=BacktestConfig(
            initial_aum=args.initial_aum,
            sigma_target=args.sigma_target,
            lev_cap=args.lev_cap,
            commission_per_share=args.commission_per_share,
            slippage_per_share=args.slippage_per_share,
            decision_freq_mins=args.decision_freq_mins,
            first_trade_time=args.first_trade_time,
            use_next_bar_open=args.use_next_bar_open,
            minute_stop_monitoring=args.minute_stop_monitoring,
            spread_bps=args.spread_bps,
        ),
    )
    result = engine.run(bars)

    result.equity_curve.to_parquet(p["baseline_equity"], index=False)
    result.trades.to_parquet(p["baseline_trades"], index=False)
    _dump_json(p["baseline_summary"], result.summary)

    logger.info("Saved baseline equity to %s", p["baseline_equity"])
    logger.info("Saved baseline trades to %s", p["baseline_trades"])
    logger.info("Saved baseline summary to %s", p["baseline_summary"])


def cmd_run_system_ml_backtest_engine(args: argparse.Namespace) -> None:
    p = _paths()
    bars = _read_parquet_required(p["enriched"])
    feat = bars

    model, calibrator, threshold, prob_q20, prob_q40, prob_q60, prob_q80 = _load_artifacts(
        args.model_path,
        args.calibration_path,
        args.threshold_path,
    )

    strategy = BaselineNoiseAreaStrategy(
        decision_freq_mins=args.decision_freq_mins,
        first_trade_time=args.first_trade_time,
    )
    risk_manager = MLOutputSizerRiskManager(
        threshold=threshold,
        prob_q20=prob_q20,
        prob_q40=prob_q40,
        prob_q60=prob_q60,
        prob_q80=prob_q80,
        allocation_mode="soft_size",
        neutral_zone=not args.no_neutral_zone,
        size_floor=args.size_floor,
        size_cap=args.size_cap,
        regime_overlay=not args.no_regime_overlay,
        regime_lookback_months=args.regime_lookback_months,
        regime_min_trades=args.regime_min_trades,
    )

    def _ml_market_state_builder(**kwargs: Any) -> dict[str, Any]:
        row = kwargs["row"]
        signal = kwargs["signal"]
        if int(signal.desired_side) == 0:
            return {}
        x_row = _build_feature_row(row, side=int(signal.desired_side), model=model)
        if x_row.isna().any(axis=None):
            return {"p_good": float("nan")}
        scores = _raw_scores(model, x_row)
        p_good = float(calibrator.predict_proba(scores)[0])
        return {"p_good": p_good}

    engine = BacktestEngine(
        strategy=strategy,
        risk_manager=risk_manager,
        config=BacktestConfig(
            initial_aum=args.initial_aum,
            sigma_target=args.sigma_target,
            lev_cap=args.lev_cap,
            commission_per_share=args.commission_per_share,
            slippage_per_share=args.slippage_per_share,
            decision_freq_mins=args.decision_freq_mins,
            first_trade_time=args.first_trade_time,
            use_next_bar_open=args.use_next_bar_open,
            minute_stop_monitoring=args.minute_stop_monitoring,
            spread_bps=args.spread_bps,
        ),
        market_state_builder=_ml_market_state_builder,
    )
    result = engine.run(feat)

    baseline_summary = run_baseline_backtest(
        feat,
        initial_aum=args.initial_aum,
        sigma_target=args.sigma_target,
        lev_cap=args.lev_cap,
        commission_per_share=args.commission_per_share,
        slippage_per_share=args.slippage_per_share,
        decision_freq_mins=args.decision_freq_mins,
        first_trade_time=args.first_trade_time,
        use_next_bar_open=args.use_next_bar_open,
        minute_stop_monitoring=args.minute_stop_monitoring,
        spread_bps=args.spread_bps,
    )["summary"]

    ml_metrics = dict(result.summary)
    ml_metrics.update(
        {
            "threshold": float(threshold),
            "allocation_mode": "soft_size",
            "size_floor": float(args.size_floor),
            "size_cap": float(args.size_cap),
            "neutral_zone": bool(not args.no_neutral_zone),
            "regime_overlay": bool(not args.no_regime_overlay),
            "regime_lookback_months": int(args.regime_lookback_months),
            "regime_min_trades": int(args.regime_min_trades),
            "avg_size_mult": float(result.trades["size_mult"].dropna().mean()) if not result.trades.empty else 1.0,
            "overlay_enabled_rate": float(result.trades["overlay_enabled"].fillna(False).astype(float).mean())
            if not result.trades.empty
            else 0.0,
        }
    )

    keys = [
        "final_equity",
        "sharpe",
        "cagr_ish",
        "max_drawdown",
        "trades_count",
        "turnover",
        "total_costs",
    ]
    comparison = pd.DataFrame(
        {
            "baseline": {k: float(baseline_summary.get(k, np.nan)) for k in keys},
            "ml_filter": {k: float(ml_metrics.get(k, np.nan)) for k in keys},
        }
    )
    comparison["delta"] = comparison["ml_filter"] - comparison["baseline"]

    result.equity_curve.to_parquet(p["ml_equity"], index=False)
    result.trades.to_parquet(p["ml_trades"], index=False)
    comparison.to_parquet(p["ml_comparison"], index=True)
    _dump_json(p["ml_metrics"], ml_metrics)
    _dump_json(
        p["ml_comparison_json"],
        {
            "baseline_summary": baseline_summary,
            "ml_metrics": ml_metrics,
            "comparison": comparison.to_dict(orient="index"),
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
    p_base.add_argument("--decision-freq-mins", type=int, default=DEFAULT_DECISION_FREQ_MINS)
    p_base.add_argument("--first-trade-time", default="10:00")
    _add_realism_args(p_base)
    p_base.set_defaults(func=cmd_baseline_backtest)

    p_ds = sub.add_parser("build_ml_dataset", help="Build ML dataset from baseline entries")
    p_ds.add_argument("--initial-aum", type=float, default=100000)
    p_ds.add_argument("--sigma-target", type=float, default=0.02)
    p_ds.add_argument("--lev-cap", type=float, default=4.0)
    p_ds.add_argument("--commission-per-share", type=float, default=0.0035)
    p_ds.add_argument("--slippage-per-share", type=float, default=0.001)
    p_ds.add_argument("--decision-freq-mins", type=int, default=DEFAULT_DECISION_FREQ_MINS)
    p_ds.add_argument("--first-trade-time", default="10:00")
    p_ds.add_argument(
        "--label-mode",
        choices=["baseline_trade", "fixed_horizon"],
        default=DEFAULT_SCOREFORWARD_LABEL_MODE,
    )
    p_ds.add_argument("--horizon-mins", type=int, default=DEFAULT_ML_LABEL_HORIZON_MINS)
    _add_realism_args(p_ds)
    p_ds.set_defaults(func=cmd_build_ml_dataset)

    p_train = sub.add_parser("train_ml", help="Train walk-forward ML models")
    p_train.add_argument("--train-months", type=int, default=24)
    p_train.add_argument("--val-months", type=int, default=3)
    p_train.add_argument("--test-months", type=int, default=3)
    p_train.add_argument("--step-months", type=int, default=3)
    p_train.add_argument("--target-mode", choices=["binary", "large_winner"], default="binary")
    p_train.add_argument("--target-quantile", type=float, default=0.70)
    p_train.add_argument("--side-filter", choices=["all", "long", "short"], default="all")
    p_train.add_argument("--artifact-subdir", default="models")
    p_train.set_defaults(func=cmd_train_ml)

    p_bt_ml = sub.add_parser("backtest_ml", help="Run ML-filtered backtest")
    p_bt_ml.add_argument("--initial-aum", type=float, default=100000)
    p_bt_ml.add_argument("--sigma-target", type=float, default=0.02)
    p_bt_ml.add_argument("--lev-cap", type=float, default=4.0)
    p_bt_ml.add_argument("--commission-per-share", type=float, default=0.0035)
    p_bt_ml.add_argument("--slippage-per-share", type=float, default=0.001)
    p_bt_ml.add_argument("--decision-freq-mins", type=int, default=DEFAULT_DECISION_FREQ_MINS)
    p_bt_ml.add_argument("--first-trade-time", default="10:00")
    p_bt_ml.add_argument("--model-path", default=None)
    p_bt_ml.add_argument("--calibration-path", default=None)
    p_bt_ml.add_argument("--threshold-path", default=None)
    p_bt_ml.add_argument("--size-floor", type=float, default=DEFAULT_SOFT_SIZE_FLOOR)
    p_bt_ml.add_argument("--size-cap", type=float, default=DEFAULT_SOFT_SIZE_CAP)
    p_bt_ml.add_argument("--market-vol-overlay", action="store_true")
    p_bt_ml.add_argument("--market-vol-lookback-days", type=int, default=DEFAULT_MM_LOOKBACK_DAYS)
    p_bt_ml.add_argument("--market-vol-floor", type=float, default=DEFAULT_MM_FLOOR)
    p_bt_ml.add_argument("--market-vol-cap", type=float, default=DEFAULT_MM_CAP)
    _add_realism_args(p_bt_ml)
    _add_realistic_soft_improvement_args(p_bt_ml)
    p_bt_ml.set_defaults(func=cmd_backtest_ml)

    p_bt_mlsf = sub.add_parser("backtest_ml_scoreforward", help="Run true rolling retrain / score-forward ML backtest")
    p_bt_mlsf.add_argument("--methods", default="soft", help="Comma-separated serious methods: soft,mm")
    p_bt_mlsf.add_argument("--initial-aum", type=float, default=100000)
    p_bt_mlsf.add_argument("--sigma-target", type=float, default=0.02)
    p_bt_mlsf.add_argument("--lev-cap", type=float, default=4.0)
    p_bt_mlsf.add_argument("--commission-per-share", type=float, default=0.0035)
    p_bt_mlsf.add_argument("--slippage-per-share", type=float, default=0.001)
    p_bt_mlsf.add_argument("--decision-freq-mins", type=int, default=DEFAULT_DECISION_FREQ_MINS)
    p_bt_mlsf.add_argument("--first-trade-time", default="10:00")
    p_bt_mlsf.add_argument(
        "--label-mode",
        choices=["baseline_trade", "fixed_horizon"],
        default=DEFAULT_SCOREFORWARD_LABEL_MODE,
    )
    p_bt_mlsf.add_argument("--horizon-mins", type=int, default=DEFAULT_ML_LABEL_HORIZON_MINS)
    p_bt_mlsf.add_argument("--train-months", type=int, default=24)
    p_bt_mlsf.add_argument("--val-months", type=int, default=3)
    p_bt_mlsf.add_argument("--test-months", type=int, default=3)
    p_bt_mlsf.add_argument("--step-months", type=int, default=3)
    p_bt_mlsf.add_argument("--warmup-days", type=int, default=90)
    p_bt_mlsf.add_argument("--market-vol-lookback-days", type=int, default=DEFAULT_MM_LOOKBACK_DAYS)
    p_bt_mlsf.add_argument("--market-vol-floor", type=float, default=DEFAULT_MM_FLOOR)
    p_bt_mlsf.add_argument("--market-vol-cap", type=float, default=DEFAULT_MM_CAP)
    _add_realism_args(p_bt_mlsf)
    _add_realistic_soft_improvement_args(p_bt_mlsf)
    p_bt_mlsf.set_defaults(func=cmd_backtest_ml_scoreforward)

    p_live_board = sub.add_parser("live_strategy_board", help="Stream live Alpaca bars and evaluate baseline + top realistic strategies")
    p_live_board.add_argument("--symbol", default="SPY")
    p_live_board.add_argument("--feed", default=None, help="Alpaca live feed name, for example iex or sip")
    p_live_board.add_argument("--variants", default=",".join(list_live_variants()))
    p_live_board.add_argument("--history-business-days", type=int, default=20)
    p_live_board.add_argument("--refresh-seconds", type=float, default=5.0)
    p_live_board.add_argument("--duration-seconds", type=float, default=300.0)
    p_live_board.add_argument("--with-account", action="store_true", help="Also fetch paper account/position state while monitoring")
    p_live_board.add_argument("--force-seed", action="store_true", help="Force-refresh the historical warmup bars before streaming")
    p_live_board.set_defaults(func=cmd_live_strategy_board)

    p_live_paper = sub.add_parser("live_paper_strategy", help="Run one live websocket-driven paper strategy")
    p_live_paper.add_argument("--variant", choices=list_live_variants(), default="soft_hybrid_7_5")
    p_live_paper.add_argument("--symbol", default="SPY")
    p_live_paper.add_argument("--feed", default=None, help="Alpaca live feed name, for example iex or sip")
    p_live_paper.add_argument("--history-business-days", type=int, default=20)
    p_live_paper.add_argument("--refresh-seconds", type=float, default=5.0)
    p_live_paper.add_argument("--duration-seconds", type=float, default=300.0)
    p_live_paper.add_argument("--sigma-target", type=float, default=0.02)
    p_live_paper.add_argument("--lev-cap", type=float, default=4.0)
    p_live_paper.add_argument("--dry-run", action="store_true", help="Compute actions without routing paper orders")
    p_live_paper.add_argument("--allow-live", action="store_true", help="Override the paper-only broker safety check")
    p_live_paper.add_argument("--force-seed", action="store_true", help="Force-refresh the historical warmup bars before streaming")
    p_live_paper.set_defaults(func=cmd_live_paper_strategy)

    p_run = sub.add_parser("run_system", help="Run the modular engine system")
    run_sub = p_run.add_subparsers(dest="run_system_command", required=True)

    p_run_base = run_sub.add_parser("baseline_backtest_engine", help="Run baseline through BacktestEngine")
    p_run_base.add_argument("--initial-aum", type=float, default=100000)
    p_run_base.add_argument("--sigma-target", type=float, default=0.02)
    p_run_base.add_argument("--lev-cap", type=float, default=4.0)
    p_run_base.add_argument("--commission-per-share", type=float, default=0.0035)
    p_run_base.add_argument("--slippage-per-share", type=float, default=0.001)
    p_run_base.add_argument("--decision-freq-mins", type=int, default=DEFAULT_DECISION_FREQ_MINS)
    p_run_base.add_argument("--first-trade-time", default="10:00")
    _add_realism_args(p_run_base)
    p_run_base.set_defaults(func=cmd_run_system_baseline_engine)

    p_run_ml = run_sub.add_parser("ml_backtest_engine", help="Run ML-sized baseline through BacktestEngine")
    p_run_ml.add_argument("--initial-aum", type=float, default=100000)
    p_run_ml.add_argument("--sigma-target", type=float, default=0.02)
    p_run_ml.add_argument("--lev-cap", type=float, default=4.0)
    p_run_ml.add_argument("--commission-per-share", type=float, default=0.0035)
    p_run_ml.add_argument("--slippage-per-share", type=float, default=0.001)
    p_run_ml.add_argument("--decision-freq-mins", type=int, default=DEFAULT_DECISION_FREQ_MINS)
    p_run_ml.add_argument("--first-trade-time", default="10:00")
    p_run_ml.add_argument("--model-path", default=None)
    p_run_ml.add_argument("--calibration-path", default=None)
    p_run_ml.add_argument("--threshold-path", default=None)
    p_run_ml.add_argument("--size-floor", type=float, default=DEFAULT_SOFT_SIZE_FLOOR)
    p_run_ml.add_argument("--size-cap", type=float, default=DEFAULT_SOFT_SIZE_CAP)
    p_run_ml.add_argument("--no-neutral-zone", action="store_true")
    p_run_ml.add_argument("--no-regime-overlay", action="store_true")
    p_run_ml.add_argument("--regime-lookback-months", type=int, default=6)
    p_run_ml.add_argument("--regime-min-trades", type=int, default=80)
    _add_realism_args(p_run_ml)
    p_run_ml.set_defaults(func=cmd_run_system_ml_backtest_engine)

    return parser


def main() -> None:
    """Run CLI command dispatch."""
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
