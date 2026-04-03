"""Backtesting logic that applies an ML probability filter to baseline entries/flips."""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .baseline_strategy import (
    OpenTrade,
    _build_decision_time_set,
    compute_breakout_margin,
    compute_break_strength,
    compute_scalein_target_shares,
    flip_allowed_by_hysteresis,
    get_execution_row,
    apply_execution_spread,
    stop_trigger_details,
    trend_signal_still_valid,
    _compute_daily_sizing_table,
    _desired_direction,
    _normalize_df,
    _stop_triggered,
    run_baseline_backtest,
)
from .config import load_config
from .core.types import Signal
from .features_ml import _compute_feature_frame
from .metrics import summarize_backtest
from .ml_overlay_robust import (
    compute_overlay_enabled_flag,
    convex_rank_bucket_map,
    execution_aware_entry_multiplier,
    execution_aware_relative_entry_multiplier,
    fast_alpha_tactical_multiplier,
    intraday_risk_size_multiplier,
    market_vol_managed_multiplier,
    panic_derisk_multiplier,
    risk_state_multiplier,
    rolling_rank_percentile,
    strategy_vol_managed_multiplier,
    shrink_toward_one,
    size_map_with_neutral_zone,
    trend_state_multiplier,
)
from .strategies import MLOutputSizerRiskManager

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "signed_break_distance",
    "break_strength",
    "band_width",
    "vwap_diff",
    "intraday_return",
    "ret_30m",
    "realized_vol_30m",
    "whipsaw_60m",
    "time_of_day_minutes",
    "tod_sin",
    "tod_cos",
]


@dataclass(frozen=True)
class ClosedTradeResult:
    """Pure bookkeeping payload for a realized exit."""

    pnl_delta: float
    costs_delta: float
    notional_delta: float
    trade_record: dict[str, Any]
    history_record: dict[str, Any]


def _expected_feature_names(model: Any) -> list[str] | None:
    expected: list[str] | None = None
    if hasattr(model, "feature_names_in_"):
        expected = [str(c) for c in getattr(model, "feature_names_in_")]
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf") if hasattr(model.named_steps, "get") else None
        if clf is not None and hasattr(clf, "feature_names_in_"):
            expected = [str(c) for c in getattr(clf, "feature_names_in_")]
    return expected


def apply_ml_filter(signals: pd.Series, probas: pd.Series, threshold: float) -> pd.Series:
    """Filter baseline signal proposals with probability threshold."""
    s = pd.Series(signals)
    p = pd.Series(probas).reindex(s.index)
    return s.where(p >= threshold, other=0)


def _raw_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    proba = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    eps = 1e-8
    proba = np.clip(proba, eps, 1 - eps)
    return np.log(proba / (1 - proba))


def _load_artifacts(
    model_path: str | Path | None,
    calibration_path: str | Path | None,
    threshold_path: str | Path | None,
) -> tuple[Any, Any, float, float | None, float | None, float | None, float | None]:
    config = load_config()
    model_dir = config.data_dir / "models"

    m_path = Path(model_path) if model_path is not None else model_dir / "best_model.joblib"
    c_path = Path(calibration_path) if calibration_path is not None else model_dir / "calibration.joblib"
    t_path = Path(threshold_path) if threshold_path is not None else model_dir / "selected_threshold.json"

    if not m_path.exists() or not c_path.exists() or not t_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts. Expected: {m_path}, {c_path}, {t_path}"
        )

    model = joblib.load(m_path)
    calibrator = joblib.load(c_path)
    with t_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
        threshold = float(payload["threshold"])
        prob_q20 = float(payload["prob_q20"]) if "prob_q20" in payload else None
        prob_q40 = float(payload["prob_q40"]) if "prob_q40" in payload else None
        prob_q60 = float(payload["prob_q60"]) if "prob_q60" in payload else None
        prob_q80 = float(payload["prob_q80"]) if "prob_q80" in payload else None

    logger.info("Loaded model artifacts from %s", model_dir)
    return model, calibrator, threshold, prob_q20, prob_q40, prob_q60, prob_q80


def _load_artifact_bundle(
    model_path: str | Path | None,
    calibration_path: str | Path | None,
    threshold_path: str | Path | None,
) -> dict[str, Any]:
    model, calibrator, threshold, prob_q20, prob_q40, prob_q60, prob_q80 = _load_artifacts(
        model_path,
        calibration_path,
        threshold_path,
    )
    return {
        "model": model,
        "calibrator": calibrator,
        "threshold": threshold,
        "prob_q20": prob_q20,
        "prob_q40": prob_q40,
        "prob_q60": prob_q60,
        "prob_q80": prob_q80,
    }


def _align_model_features(model: Any, X_row: pd.DataFrame) -> pd.DataFrame:
    """Align runtime features to the columns expected by the fitted model."""
    expected = _expected_feature_names(model)
    if not expected:
        return X_row

    aligned = X_row.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[expected]


def _build_feature_row(row: pd.Series, side: int, model: Any | None = None) -> pd.DataFrame:
    signed_break = (
        (float(row["close"]) - float(row["UB"])) / float(row["UB"])
        if side > 0
        else (float(row["LB"]) - float(row["close"])) / float(row["LB"])
    )

    values = {
        "signed_break_distance": signed_break,
        "break_strength": float(row["break_strength"]),
        "band_width": float(row["band_width"]),
        "vwap_diff": float(row["vwap_diff"]),
        "intraday_return": float(row["intraday_return"]),
        "ret_30m": float(row["ret_30m"]),
        "realized_vol_30m": float(row["realized_vol_30m"]),
        "whipsaw_60m": float(row["whipsaw_60m"]),
        "time_of_day_minutes": float(row["time_of_day_minutes"]),
        "tod_sin": float(row["tod_sin"]),
        "tod_cos": float(row["tod_cos"]),
    }
    X_row = pd.DataFrame([values], columns=FEATURE_COLS)
    expected = _expected_feature_names(model) if model is not None else None
    if expected:
        symbol = str(row.get("symbol", ""))
        symbol_cols = [col for col in expected if col.startswith("symbol_")]
        for col in symbol_cols:
            X_row[col] = 1.0 if col == f"symbol_{symbol}" else 0.0
    return X_row


def _summary_with_extras(
    equity_curve: pd.DataFrame,
    trades_df: pd.DataFrame,
    total_notional_traded: float,
    initial_aum: float,
) -> dict[str, float]:
    summary = summarize_backtest(
        equity_curve["daily_return"] if not equity_curve.empty else pd.Series(dtype=float)
    )
    summary.update(
        {
            "final_equity": float(equity_curve["equity"].iloc[-1])
            if not equity_curve.empty
            else float(initial_aum),
            "trades_count": int(len(trades_df)),
            "turnover": float(total_notional_traded / equity_curve["equity"].mean())
            if not equity_curve.empty and float(equity_curve["equity"].mean()) != 0
            else 0.0,
            "total_costs": float(trades_df["costs"].sum()) if not trades_df.empty else 0.0,
        }
    )
    return summary


def _first_stop_hit(
    day_df: pd.DataFrame,
    trade: OpenTrade,
    *,
    start_idx: int,
    end_idx: int,
    minute_aware: bool,
    stop_mode: str = "minute_full",
    catastrophic_stop_bps: float = 0.0,
) -> tuple[int, pd.Series, float] | None:
    """Return the first stop breach between two intraday row indices."""
    if int(end_idx) < int(start_idx):
        return None
    for scan_idx in range(int(start_idx), int(end_idx) + 1):
        scan_row = day_df.iloc[int(scan_idx)]
        if stop_mode == "hybrid":
            stop_hit, raw_exit_price = _hybrid_stop_trigger_details(
                trade,
                scan_row,
                catastrophic_stop_bps=catastrophic_stop_bps,
            )
        else:
            stop_hit, raw_exit_price = stop_trigger_details(
                trade,
                scan_row,
                minute_aware=minute_aware,
            )
        if stop_hit:
            return int(scan_idx), scan_row, float(raw_exit_price)
    return None


def _hybrid_stop_trigger_details(
    trade: OpenTrade,
    row: pd.Series,
    *,
    catastrophic_stop_bps: float,
) -> tuple[bool, float]:
    """Return only catastrophic minute-stop breaches between decision times.

    The standard stop is still checked at decision timestamps. Between
    decisions, this hybrid mode exits only if price overshoots the stop by a
    configurable amount. The goal is to keep minute-level catastrophe
    protection while reducing churn from marginal stop touches.
    """
    stop_bps = max(float(catastrophic_stop_bps), 0.0)
    if int(trade.side) > 0:
        stop_level = max(float(row["UB"]), float(row["VWAP"]))
        catastrophic_level = stop_level * (1.0 - stop_bps / 10000.0)
        open_px = float(row.get("open", row["close"]))
        low_px = float(row.get("low", row["close"]))
        if open_px <= catastrophic_level:
            return True, open_px
        if low_px <= catastrophic_level:
            return True, catastrophic_level
        return False, float(row["close"])

    stop_level = min(float(row["LB"]), float(row["VWAP"]))
    catastrophic_level = stop_level * (1.0 + stop_bps / 10000.0)
    open_px = float(row.get("open", row["close"]))
    high_px = float(row.get("high", row["close"]))
    if open_px >= catastrophic_level:
        return True, open_px
    if high_px >= catastrophic_level:
        return True, catastrophic_level
    return False, float(row["close"])


def _history_window_values(
    history: list[dict[str, Any]],
    current_day: str,
    symbol: str,
    key: str,
    n_days: int,
) -> list[float]:
    """Collect the prior N daily values for a symbol-specific history field."""
    days: list[str] = []
    seen: set[str] = set()
    for item in reversed(history):
        if item.get("symbol") != symbol:
            continue
        day = str(item.get("date"))
        if day >= current_day or day in seen:
            continue
        seen.add(day)
        days.append(day)
        if len(seen) >= int(n_days):
            break
    if len(seen) < int(n_days):
        return []
    keep = set(days)
    return [
        float(item[key])
        for item in history
        if item.get("symbol") == symbol
        and str(item.get("date")) in keep
        and not pd.isna(item.get(key, np.nan))
    ]


def _scale_shares(
    base_shares: int,
    size_mult: float,
    aum_prev: float,
    lev_cap: float,
    exec_px_raw: float,
) -> int:
    """Scale a base share count and enforce leverage-aware caps."""
    eff_shares = int(math.floor(base_shares * float(size_mult))) if base_shares > 0 else 0
    max_shares = int(math.floor(aum_prev * lev_cap / exec_px_raw)) if exec_px_raw > 0 else 0
    if max_shares > 0:
        eff_shares = min(eff_shares, max_shares)
    if eff_shares <= 0 and base_shares > 0:
        eff_shares = 1 if (max_shares > 0 or exec_px_raw <= 0) else 0
    return eff_shares


def _apply_size_multiplier(
    size_mult: float,
    overlay_mult: float,
    base_shares: int,
    aum_prev: float,
    lev_cap: float,
    exec_px_raw: float,
) -> tuple[float, int]:
    """Apply one overlay multiplier and recompute capped share size."""
    updated_mult = float(size_mult) * float(overlay_mult)
    return updated_mult, _scale_shares(base_shares, updated_mult, aum_prev, lev_cap, exec_px_raw)


def _close_trade_result(
    open_trade: OpenTrade,
    *,
    exit_timestamp: pd.Timestamp,
    exit_price: float,
    cost_per_share: float,
) -> ClosedTradeResult:
    """Build the common exit bookkeeping payload for a realized trade."""
    exit_cost = cost_per_share * open_trade.shares
    gross = open_trade.side * open_trade.shares * (exit_price - open_trade.entry_price)
    realized_pnl = gross - open_trade.entry_cost - exit_cost
    trade_record = {
        "decision_timestamp": getattr(open_trade, "decision_timestamp", open_trade.entry_timestamp),
        "entry_timestamp": open_trade.entry_timestamp,
        "exit_timestamp": exit_timestamp,
        "side": "long" if open_trade.side > 0 else "short",
        "shares": open_trade.shares,
        "entry_price": open_trade.entry_price,
        "exit_price": exit_price,
        "pnl": realized_pnl,
        "costs": open_trade.entry_cost + exit_cost,
        "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
        "p_rank": getattr(open_trade, "p_rank", np.nan),
        "size_mult": getattr(open_trade, "size_mult", np.nan),
        "overlay_enabled": getattr(open_trade, "overlay_enabled", np.nan),
        "risk_value": getattr(open_trade, "risk_value", np.nan),
        "used_floor": getattr(open_trade, "used_floor", np.nan),
        "used_cap": getattr(open_trade, "used_cap", np.nan),
        "strategy_vol_mult": getattr(open_trade, "strategy_vol_mult", np.nan),
        "market_vol_mult": getattr(open_trade, "market_vol_mult", np.nan),
        "panic_mult": getattr(open_trade, "panic_mult", np.nan),
        "trend_state_mult": getattr(open_trade, "trend_state_mult", np.nan),
        "boost_triggered_day": bool(getattr(open_trade, "boost_triggered_day", False)),
        "scale_in_count": int(getattr(open_trade, "scale_in_count", 0)),
        "scale_in_shares": int(getattr(open_trade, "scale_in_shares", 0)),
        "action_log": list(getattr(open_trade, "action_log", ["entry"])),
    }
    return ClosedTradeResult(
        pnl_delta=gross - exit_cost,
        costs_delta=exit_cost,
        notional_delta=open_trade.shares * exit_price,
        trade_record=trade_record,
        history_record={
            "exit_timestamp": exit_timestamp,
            "pnl": realized_pnl,
            "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
        },
    )


def _build_open_trade(
    *,
    desired: int,
    shares: int,
    entry_timestamp: pd.Timestamp,
    entry_price: float,
    entry_cost: float,
    decision_timestamp: pd.Timestamp,
    p_good: float,
    p_rank: float,
    risk_value: float,
    used_floor: float,
    used_cap: float,
    size_mult: float,
    bucket_label: str,
    overlay_enabled: bool,
    regime_spread: float,
    lookback_n: int,
    strategy_vol_mult: float,
    market_vol_mult: float,
    panic_mult: float,
    trend_state_mult: float,
    fast_alpha_mult: float,
    fast_alpha_favorable: bool,
    intraday_risk_mult: float,
    intraday_risk_threshold: float,
    execution_chase_mult: float,
    entry_adverse_bps: float,
    entry_adverse_return: float,
    entry_adverse_threshold: float,
    base_shares: int,
) -> OpenTrade:
    """Create an open trade and attach ML/overlay metadata used downstream."""
    trade = OpenTrade(
        side=desired,
        shares=shares,
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        entry_cost=entry_cost,
        decision_timestamp=decision_timestamp,
    )
    setattr(trade, "entry_p_good", p_good)
    setattr(trade, "p_rank", p_rank)
    setattr(trade, "risk_value", risk_value)
    setattr(trade, "used_floor", used_floor)
    setattr(trade, "used_cap", used_cap)
    setattr(trade, "flip_blocked", False)
    setattr(trade, "base_shares", int(base_shares))
    setattr(trade, "trend_valid_count", 0)
    setattr(trade, "boost_triggered_day", False)
    setattr(trade, "scale_in_count", 0)
    setattr(trade, "scale_in_shares", 0)
    setattr(trade, "action_log", ["entry"])
    setattr(trade, "size_mult", size_mult)
    setattr(trade, "bucket_label", bucket_label)
    setattr(trade, "overlay_enabled", overlay_enabled)
    setattr(trade, "regime_spread", regime_spread)
    setattr(trade, "lookback_n", lookback_n)
    setattr(trade, "strategy_vol_mult", float(strategy_vol_mult))
    setattr(trade, "market_vol_mult", float(market_vol_mult))
    setattr(trade, "panic_mult", float(panic_mult))
    setattr(trade, "trend_state_mult", float(trend_state_mult))
    setattr(trade, "fast_alpha_mult", float(fast_alpha_mult))
    setattr(trade, "fast_alpha_favorable", bool(fast_alpha_favorable))
    setattr(trade, "intraday_risk_mult", float(intraday_risk_mult))
    setattr(trade, "intraday_risk_threshold", float(intraday_risk_threshold))
    setattr(trade, "execution_chase_mult", float(execution_chase_mult))
    setattr(trade, "entry_adverse_bps", float(entry_adverse_bps))
    setattr(trade, "entry_adverse_return", float(entry_adverse_return))
    setattr(trade, "entry_adverse_threshold", float(entry_adverse_threshold))
    return trade


def run_ml_filtered_backtest(
    df: pd.DataFrame,
    initial_aum: float = 100000,
    sigma_target: float = 0.02,
    lev_cap: float = 4.0,
    commission_per_share: float = 0.0035,
    slippage_per_share: float = 0.001,
    decision_freq_mins: int = 30,
    first_trade_time: str = "10:00",
    margin_min_bps: float = 0.0,
    flip_hysteresis_bps: float = 0.0,
    cooldown_steps: int = 0,
    cooldown_on_stop: bool = True,
    trend_scalein_enabled: bool = False,
    trend_persistence_steps: int = 2,
    trend_boost_mult: float = 1.8,
    trend_boost_cap_mult: float = 2.5,
    trend_scalein_once: bool = True,
    use_next_bar_open: bool = False,
    minute_stop_monitoring: bool = False,
    spread_bps: float = 0.0,
    break_strength_min: float | None = None,
    artifact_bundle: dict[str, Any] | None = None,
    long_artifact_bundle: dict[str, Any] | None = None,
    short_artifact_bundle: dict[str, Any] | None = None,
    model_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
    long_model_path: str | Path | None = None,
    long_calibration_path: str | Path | None = None,
    long_threshold_path: str | Path | None = None,
    short_model_path: str | Path | None = None,
    short_calibration_path: str | Path | None = None,
    short_threshold_path: str | Path | None = None,
    flip_reject_mode: str = "hold",
    filter_mode: str = "entry_only",
    allocation_mode: str = "soft_size",
    size_floor: float = 0.5,
    size_cap: float = 1.5,
    prob_q20: float | None = None,
    prob_q40: float | None = None,
    prob_q60: float | None = None,
    prob_q80: float | None = None,
    neutral_zone: bool = True,
    regime_overlay: bool = True,
    regime_lookback_months: int = 6,
    regime_min_trades: int = 80,
    rank_window_days: int = 60,
    neutral_lo: float = 0.4,
    neutral_hi: float = 0.6,
    shrink_lam: float = 0.5,
    overlay_gate_lookback_days: int = 63,
    risk_quantile: float = 0.8,
    high_risk_floor: float = 0.95,
    high_risk_cap: float = 1.05,
    low_cut: float = 0.40,
    high_cut: float = 0.80,
    low_mult: float = 0.70,
    mid_mult: float = 1.00,
    high_mult: float = 2.00,
    convex_cap_mult: float = 2.50,
    strategy_vol_overlay: bool = False,
    strategy_vol_lookback_days: int = 20,
    strategy_vol_floor: float = 0.50,
    strategy_vol_cap: float = 1.50,
    market_vol_overlay: bool = False,
    market_vol_lookback_days: int = 20,
    market_vol_floor: float = 0.50,
    market_vol_cap: float = 1.50,
    panic_derisk_overlay: bool = False,
    panic_return_lookback_days: int = 20,
    panic_vol_lookback_days: int = 20,
    panic_vol_quantile: float = 0.80,
    panic_exposure: float = 0.50,
    trend_state_overlay: bool = False,
    trend_state_lookback_days: int = 60,
    trend_state_low_exposure: float = 0.85,
    trend_state_high_exposure: float = 1.15,
    fast_alpha_overlay: bool = False,
    fast_alpha_bar_mins: int = 5,
    fast_alpha_favorable_mult: float = 1.15,
    fast_alpha_unfavorable_mult: float = 0.85,
    execution_chase_control: bool = False,
    execution_chase_bps: float = 8.0,
    execution_chase_mult: float = 0.5,
    execution_chase_relative: bool = False,
    execution_chase_band_frac: float = 0.15,
    execution_chase_vol_mult: float = 0.75,
    hybrid_stop_mode: bool = False,
    catastrophic_stop_bps: float = 10.0,
    intraday_risk_overlay: bool = False,
    intraday_risk_lookback_days: int = 60,
    intraday_risk_quantile: float = 0.8,
    intraday_risk_downsize_mult: float = 0.75,
    daily_sizing: pd.DataFrame | None = None,
    trade_start_date: str | pd.Timestamp | None = None,
) -> dict:
    """Run baseline backtest with ML probability integration.

    Default behavior is soft sizing (`allocation_mode="soft_size"`), which keeps
    baseline trade directions and scales position size by model confidence.
    """
    if break_strength_min is not None:
        warnings.warn(
            "break_strength_min is deprecated and ignored; use margin_min_bps instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if float(trend_boost_mult) > float(trend_boost_cap_mult):
        raise ValueError("trend_boost_mult must be <= trend_boost_cap_mult")
    if int(trend_persistence_steps) < 1:
        raise ValueError("trend_persistence_steps must be >= 1")

    if flip_reject_mode not in {"hold", "close_flat"}:
        raise ValueError("flip_reject_mode must be one of: 'hold', 'close_flat'")
    if filter_mode not in {"entry_only", "all_candidates"}:
        raise ValueError("filter_mode must be one of: 'entry_only', 'all_candidates'")
    if allocation_mode not in {"hard_filter", "soft_size", "robust_soft_size", "convex_rank_size"}:
        raise ValueError("allocation_mode must be one of: 'hard_filter', 'soft_size', 'robust_soft_size', 'convex_rank_size'")
    if size_floor <= 0 or size_cap <= 0 or size_floor > size_cap:
        raise ValueError("size_floor/size_cap must be positive and satisfy size_floor <= size_cap")
    if strategy_vol_floor <= 0 or strategy_vol_cap <= 0 or strategy_vol_floor > strategy_vol_cap:
        raise ValueError("strategy_vol_floor/strategy_vol_cap must be positive and satisfy strategy_vol_floor <= strategy_vol_cap")
    if market_vol_floor <= 0 or market_vol_cap <= 0 or market_vol_floor > market_vol_cap:
        raise ValueError("market_vol_floor/market_vol_cap must be positive and satisfy market_vol_floor <= market_vol_cap")
    if not (0.0 < float(panic_vol_quantile) < 1.0):
        raise ValueError("panic_vol_quantile must be in (0, 1)")
    if float(panic_exposure) <= 0:
        raise ValueError("panic_exposure must be positive")
    if float(trend_state_low_exposure) <= 0 or float(trend_state_high_exposure) <= 0:
        raise ValueError("trend_state exposures must be positive")
    if float(execution_chase_mult) <= 0:
        raise ValueError("execution_chase_mult must be positive")
    if float(execution_chase_band_frac) < 0 or float(execution_chase_vol_mult) < 0:
        raise ValueError("execution_chase_band_frac/execution_chase_vol_mult must be >= 0")
    if float(catastrophic_stop_bps) < 0:
        raise ValueError("catastrophic_stop_bps must be >= 0")
    if not (0.0 < float(intraday_risk_quantile) < 1.0):
        raise ValueError("intraday_risk_quantile must be in (0, 1)")
    if float(intraday_risk_downsize_mult) <= 0:
        raise ValueError("intraday_risk_downsize_mult must be positive")

    # Load model artifacts and construct side-aware sizers before touching the
    # time-series loop so the runtime path only deals with per-row decisions.
    default_bundle = (
        dict(artifact_bundle)
        if artifact_bundle is not None
        else _load_artifact_bundle(model_path, calibration_path, threshold_path)
    )
    long_bundle = (
        dict(long_artifact_bundle)
        if long_artifact_bundle is not None
        else _load_artifact_bundle(long_model_path, long_calibration_path, long_threshold_path)
        if any(v is not None for v in [long_model_path, long_calibration_path, long_threshold_path])
        else dict(default_bundle)
    )
    short_bundle = (
        dict(short_artifact_bundle)
        if short_artifact_bundle is not None
        else _load_artifact_bundle(short_model_path, short_calibration_path, short_threshold_path)
        if any(v is not None for v in [short_model_path, short_calibration_path, short_threshold_path])
        else dict(default_bundle)
    )
    threshold = float(default_bundle["threshold"])
    file_q20 = default_bundle["prob_q20"]
    file_q40 = default_bundle["prob_q40"]
    file_q60 = default_bundle["prob_q60"]
    file_q80 = default_bundle["prob_q80"]
    prob_q20 = file_q20 if prob_q20 is None else prob_q20
    prob_q40 = file_q40 if prob_q40 is None else prob_q40
    prob_q60 = file_q60 if prob_q60 is None else prob_q60
    prob_q80 = file_q80 if prob_q80 is None else prob_q80
    if allocation_mode == "soft_size" and (prob_q20 is None or prob_q80 is None or prob_q80 <= prob_q20):
        logger.warning(
            "Soft sizing requested but prob_q20/prob_q80 are missing or invalid. "
            "Falling back to neutral size multiplier 1.0."
        )
    sizer = MLOutputSizerRiskManager(
        threshold=threshold,
        prob_q20=prob_q20,
        prob_q40=prob_q40,
        prob_q60=prob_q60,
        prob_q80=prob_q80,
        allocation_mode=allocation_mode,
        neutral_zone=neutral_zone,
        size_floor=size_floor,
        size_cap=size_cap,
        regime_overlay=regime_overlay,
        regime_lookback_months=regime_lookback_months,
        regime_min_trades=regime_min_trades,
    ) if allocation_mode == "soft_size" else None
    long_sizer = (
        MLOutputSizerRiskManager(
            threshold=float(long_bundle["threshold"]),
            prob_q20=float(long_bundle["prob_q20"]) if long_bundle["prob_q20"] is not None else prob_q20,
            prob_q40=float(long_bundle["prob_q40"]) if long_bundle["prob_q40"] is not None else prob_q40,
            prob_q60=float(long_bundle["prob_q60"]) if long_bundle["prob_q60"] is not None else prob_q60,
            prob_q80=float(long_bundle["prob_q80"]) if long_bundle["prob_q80"] is not None else prob_q80,
            allocation_mode=allocation_mode,
            neutral_zone=neutral_zone,
            size_floor=size_floor,
            size_cap=size_cap,
            regime_overlay=regime_overlay,
            regime_lookback_months=regime_lookback_months,
            regime_min_trades=regime_min_trades,
        )
        if allocation_mode == "soft_size" else None
    )
    short_sizer = (
        MLOutputSizerRiskManager(
            threshold=float(short_bundle["threshold"]),
            prob_q20=float(short_bundle["prob_q20"]) if short_bundle["prob_q20"] is not None else prob_q20,
            prob_q40=float(short_bundle["prob_q40"]) if short_bundle["prob_q40"] is not None else prob_q40,
            prob_q60=float(short_bundle["prob_q60"]) if short_bundle["prob_q60"] is not None else prob_q60,
            prob_q80=float(short_bundle["prob_q80"]) if short_bundle["prob_q80"] is not None else prob_q80,
            allocation_mode=allocation_mode,
            neutral_zone=neutral_zone,
            size_floor=size_floor,
            size_cap=size_cap,
            regime_overlay=regime_overlay,
            regime_lookback_months=regime_lookback_months,
            regime_min_trades=regime_min_trades,
        )
        if allocation_mode == "soft_size" else None
    )

    bars = _normalize_df(df)
    feat = _compute_feature_frame(bars)
    feat["fast_alpha_ret"] = feat.groupby("date", sort=False)["close"].pct_change(int(fast_alpha_bar_mins))

    decision_times = _build_decision_time_set(decision_freq_mins, first_trade_time)
    cost_per_share = commission_per_share + slippage_per_share

    sizing = (
        daily_sizing.copy()
        if daily_sizing is not None
        else _compute_daily_sizing_table(bars, initial_aum, sigma_target, lev_cap)
    )
    if "date" not in sizing.columns:
        raise ValueError("daily_sizing must contain a 'date' column")
    sizing["date"] = pd.to_datetime(sizing["date"]).dt.strftime("%Y-%m-%d")
    sizing_idx = sizing.set_index("date")
    live_start = None if trade_start_date is None else pd.Timestamp(trade_start_date).strftime("%Y-%m-%d")

    equity_rows: list[dict] = []
    trades: list[dict] = []
    decisions_log: list[dict] = []
    size_multipliers: list[float] = []
    overlay_enabled_flags: list[bool] = []
    closed_trade_history: list[dict[str, Any]] = []
    overlay_enabled_history: list[dict[str, Any]] = []
    candidate_score_history: list[dict[str, Any]] = []
    risk_history: list[dict[str, Any]] = []
    bucket_labels: list[str] = []
    scale_in_legs: list[dict[str, Any]] = []
    strategy_base_return_history: list[float] = []
    strategy_vol_overlay_history: list[dict[str, Any]] = []
    market_return_history: list[float] = []
    market_vol_overlay_history: list[dict[str, Any]] = []
    prev_day_close_for_market: float | None = None
    close_history: list[float] = []
    panic_overlay_history: list[dict[str, Any]] = []
    trend_state_overlay_history: list[dict[str, Any]] = []
    fast_alpha_history: list[dict[str, Any]] = []
    execution_chase_history: list[dict[str, Any]] = []
    intraday_risk_overlay_history: list[dict[str, Any]] = []
    catastrophic_stop_events = 0

    base_floor_robust = float(size_floor)
    base_cap_robust = float(size_cap)
    if allocation_mode == "robust_soft_size" and base_floor_robust == 0.5 and base_cap_robust == 1.5:
        base_floor_robust = 0.85
        base_cap_robust = 1.15

    review_every_days = 21
    current_overlay_enabled = True
    next_overlay_review_idx = 0

    aum_prev = float(initial_aum)
    total_notional_traded = 0.0

    # The day loop is intentionally stateful: realistic execution, stop
    # scanning, and ML sizing all depend on prior realized trades.
    for day_idx, (day, day_df) in enumerate(feat.groupby("date", sort=True)):
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        is_live_day = live_start is None or str(day) >= live_start
        if allocation_mode == "robust_soft_size" and day_idx >= next_overlay_review_idx:
            recent_df = pd.DataFrame(closed_trade_history)
            current_overlay_enabled = compute_overlay_enabled_flag(
                recent_df,
                score_col="entry_p_good",
                pnl_col="pnl",
                lookback_days=overlay_gate_lookback_days,
            )
            overlay_enabled_history.append({
                "date": day,
                "overlay_enabled": bool(current_overlay_enabled),
            })
            next_overlay_review_idx = day_idx + review_every_days
        cooldown_until_ts: pd.Timestamp | None = None
        day_boost_used = False
        day_open = float(day_df.iloc[0]["open"])
        strategy_vol_mult = 1.0
        strategy_target_vol = float("nan")
        strategy_current_vol = float("nan")
        if strategy_vol_overlay:
            strategy_vol_mult, strategy_target_vol, strategy_current_vol = strategy_vol_managed_multiplier(
                strategy_base_return_history,
                lookback_days=strategy_vol_lookback_days,
                floor=strategy_vol_floor,
                cap=strategy_vol_cap,
            )
        market_vol_mult = 1.0
        market_target_vol = float("nan")
        market_current_vol = float("nan")
        if market_vol_overlay:
            market_vol_mult, market_target_vol, market_current_vol = market_vol_managed_multiplier(
                market_return_history,
                lookback_days=market_vol_lookback_days,
                floor=market_vol_floor,
                cap=market_vol_cap,
            )
        panic_mult = 1.0
        panic_ret = float("nan")
        panic_current_vol = float("nan")
        panic_threshold = float("nan")
        if panic_derisk_overlay:
            panic_mult, panic_ret, panic_current_vol, panic_threshold = panic_derisk_multiplier(
                close_history,
                market_return_history,
                return_lookback_days=panic_return_lookback_days,
                vol_lookback_days=panic_vol_lookback_days,
                vol_quantile=panic_vol_quantile,
                panic_exposure=panic_exposure,
            )
        trend_state_mult = 1.0
        trend_state_ret = float("nan")
        trend_state_threshold = float("nan")
        if trend_state_overlay:
            trend_state_mult, trend_state_ret, trend_state_threshold = trend_state_multiplier(
                close_history,
                lookback_days=trend_state_lookback_days,
                low_exposure=trend_state_low_exposure,
                high_exposure=trend_state_high_exposure,
            )

        sigma_spy = float(sizing_idx.loc[day, "sigma_spy"]) if day in sizing_idx.index else float("nan")
        leverage = float(sizing_idx.loc[day, "leverage"]) if day in sizing_idx.index else 1.0
        shares = int(math.floor((aum_prev * leverage) / day_open)) if day_open > 0 else 0

        if not is_live_day:
            day_close = float(day_df.iloc[-1]["close"])
            if prev_day_close_for_market is not None and prev_day_close_for_market > 0:
                market_return_history.append(float(day_close / prev_day_close_for_market - 1.0))
            prev_day_close_for_market = day_close
            close_history.append(day_close)
            continue

        day_pnl = 0.0
        day_costs = 0.0
        open_trade: OpenTrade | None = None

        decision_indices = day_df.index[day_df["timestamp"].dt.time.isin(decision_times)].tolist()
        stop_scan_start_idx = 0

        for decision_idx in decision_indices:
            row = day_df.iloc[decision_idx]
            decision_px = float(row["close"])
            ts = row["timestamp"]
            next_stop_scan_idx = stop_scan_start_idx
            intraday_risk_mult = 1.0
            intraday_risk_threshold = float("nan")
            execution_chase_mult_applied = 1.0
            entry_adverse_bps = 0.0
            entry_adverse_return = 0.0
            entry_adverse_threshold = 0.0

            if open_trade is not None:
                if minute_stop_monitoring:
                    scan_end_idx = decision_idx - 1 if hybrid_stop_mode else decision_idx
                    stop_hit = _first_stop_hit(
                        day_df,
                        open_trade,
                        start_idx=stop_scan_start_idx,
                        end_idx=scan_end_idx,
                        minute_aware=True,
                        stop_mode="hybrid" if hybrid_stop_mode else "minute_full",
                        catastrophic_stop_bps=catastrophic_stop_bps,
                    )
                    if stop_hit is not None:
                        hit_idx, stop_row, raw_exit_px = stop_hit
                        if hybrid_stop_mode:
                            catastrophic_stop_events += 1
                        stop_ts = stop_row["timestamp"]
                        exit_px = apply_execution_spread(raw_exit_px, -open_trade.side, spread_bps)
                        exit_result = _close_trade_result(
                            open_trade,
                            exit_timestamp=stop_ts,
                            exit_price=exit_px,
                            cost_per_share=cost_per_share,
                        )
                        day_pnl += exit_result.pnl_delta
                        day_costs += exit_result.costs_delta
                        total_notional_traded += exit_result.notional_delta
                        trades.append(
                            exit_result.trade_record
                            | {
                                "action_type": "exit",
                                "exit_reason": "stop",
                            }
                        )
                        closed_trade_history.append(exit_result.history_record)
                        if cooldown_on_stop and cooldown_steps > 0:
                            cooldown_until_ts = stop_ts + pd.Timedelta(minutes=decision_freq_mins * cooldown_steps)
                        open_trade = None
                        next_stop_scan_idx = hit_idx + 1
                    elif hybrid_stop_mode and _stop_triggered(open_trade, row):
                        exec_idx, exec_row, exec_price_field = get_execution_row(
                            day_df,
                            decision_idx,
                            use_next_bar_open=use_next_bar_open,
                        )
                        exec_px_raw = float(exec_row[exec_price_field])
                        exec_ts = exec_row["timestamp"]
                        exit_px = apply_execution_spread(exec_px_raw, -open_trade.side, spread_bps)
                        exit_result = _close_trade_result(
                            open_trade,
                            exit_timestamp=exec_ts,
                            exit_price=exit_px,
                            cost_per_share=cost_per_share,
                        )
                        day_pnl += exit_result.pnl_delta
                        day_costs += exit_result.costs_delta
                        total_notional_traded += exit_result.notional_delta
                        trades.append(
                            exit_result.trade_record
                            | {
                                "action_type": "exit",
                                "exit_reason": "stop",
                            }
                        )
                        closed_trade_history.append(exit_result.history_record)
                        if cooldown_on_stop and cooldown_steps > 0:
                            cooldown_until_ts = exec_ts + pd.Timedelta(minutes=decision_freq_mins * cooldown_steps)
                        open_trade = None
                        next_stop_scan_idx = exec_idx
                    else:
                        next_stop_scan_idx = max(next_stop_scan_idx, decision_idx + 1)
                elif _stop_triggered(open_trade, row):
                    exit_px = apply_execution_spread(float(row["close"]), -open_trade.side, spread_bps)
                    exit_result = _close_trade_result(
                        open_trade,
                        exit_timestamp=ts,
                        exit_price=exit_px,
                        cost_per_share=cost_per_share,
                    )
                    day_pnl += exit_result.pnl_delta
                    day_costs += exit_result.costs_delta
                    total_notional_traded += exit_result.notional_delta
                    trades.append(
                        exit_result.trade_record
                        | {
                            "action_type": "exit",
                            "exit_reason": "stop",
                        }
                    )
                    closed_trade_history.append(exit_result.history_record)
                    if cooldown_on_stop and cooldown_steps > 0:
                        cooldown_until_ts = ts + pd.Timedelta(minutes=decision_freq_mins * cooldown_steps)
                    open_trade = None

            desired = _desired_direction(row)
            current = 0 if open_trade is None else open_trade.side

            if desired != current:
                is_candidate = desired != 0

                p_good = np.nan
                p_rank = np.nan
                risk_value = float(row.get("band_width", np.nan))
                used_floor = np.nan
                used_cap = np.nan
                size_mult = 1.0
                bucket_label = "mid"
                overlay_enabled = True
                regime_spread = np.nan
                lookback_n = 0
                fast_alpha_mult = 1.0
                fast_alpha_favorable = False
                take_new_position = desired != 0
                gate_applied = False
                threshold = float(default_bundle["threshold"])
                symbol = str(row.get("symbol", "SPY"))

                if is_candidate:
                    gate_applied = filter_mode == "all_candidates" or current == 0

                    side_bundle = long_bundle if desired > 0 else short_bundle
                    X_row = _build_feature_row(row, side=desired, model=side_bundle["model"])
                    if not X_row.isna().any(axis=None):
                        side_model = side_bundle["model"]
                        side_calibrator = side_bundle["calibrator"]
                        threshold = float(side_bundle["threshold"])
                        X_row = _align_model_features(side_model, X_row)
                        scores = _raw_scores(side_model, X_row)
                        p_good = float(side_calibrator.predict_proba(scores)[0])

                    if allocation_mode in {"robust_soft_size", "convex_rank_size"}:
                        prior_scores = _history_window_values(candidate_score_history, day, symbol, "p_good", rank_window_days)
                        if len(prior_scores) >= 1 and not np.isnan(p_good):
                            arr = np.asarray(prior_scores, dtype=float)
                            p_rank = float((np.sum(arr < float(p_good)) + 0.5 * np.sum(arr == float(p_good))) / len(arr))
                    if allocation_mode == "robust_soft_size":
                        prior_risk = _history_window_values(risk_history, day, symbol, "risk_value", rank_window_days)
                        risk_threshold = float(np.quantile(prior_risk, risk_quantile)) if len(prior_risk) > 0 else float("nan")
                        used_floor, used_cap = risk_state_multiplier(
                            risk_value,
                            risk_threshold,
                            base_floor_robust,
                            base_cap_robust,
                            high_risk_floor,
                            high_risk_cap,
                        )
                    if allocation_mode == "hard_filter" and gate_applied:
                        take_new_position = (not np.isnan(p_good)) and (p_good >= threshold)
                    else:
                        take_new_position = True

                allow_open = True
                flip_blocked = False
                if desired != 0 and current == 0:
                    margin_min = float(margin_min_bps) / 10000.0
                    if compute_breakout_margin(row, desired) < margin_min:
                        allow_open = False
                if desired != 0 and current != 0 and desired == -current:
                    allow_open = flip_allowed_by_hysteresis(
                        row,
                        current_side=current,
                        desired_side=desired,
                        flip_hysteresis_bps=float(flip_hysteresis_bps),
                    )
                    flip_blocked = not allow_open
                    if flip_blocked and open_trade is not None:
                        setattr(open_trade, "flip_blocked", True)
                if desired != 0 and cooldown_until_ts is not None and ts < cooldown_until_ts:
                    allow_open = False

                should_close_existing = False
                blocked_flip_exit = False
                if open_trade is not None:
                    if desired == 0:
                        should_close_existing = True
                    elif take_new_position and allow_open:
                        should_close_existing = True
                    elif flip_blocked:
                        should_close_existing = True
                        blocked_flip_exit = True
                    elif flip_reject_mode == "close_flat" and take_new_position and not allow_open:
                        should_close_existing = True
                    elif flip_reject_mode == "close_flat" and not take_new_position:
                        should_close_existing = True

                exec_idx, exec_row, exec_price_field = get_execution_row(
                    day_df,
                    decision_idx,
                    use_next_bar_open=use_next_bar_open,
                )
                exec_px_raw = float(exec_row[exec_price_field])
                exec_ts = exec_row["timestamp"]

                if should_close_existing and open_trade is not None:
                    exit_px = apply_execution_spread(exec_px_raw, -open_trade.side, spread_bps)
                    exit_result = _close_trade_result(
                        open_trade,
                        exit_timestamp=exec_ts,
                        exit_price=exit_px,
                        cost_per_share=cost_per_share,
                    )
                    day_pnl += exit_result.pnl_delta
                    day_costs += exit_result.costs_delta
                    total_notional_traded += exit_result.notional_delta
                    if desired == 0:
                        exit_reason = "flat"
                    elif blocked_flip_exit:
                        exit_reason = "flip_blocked_exit"
                    else:
                        exit_reason = "flip"
                    trades.append(
                        exit_result.trade_record
                        | {
                            "action_type": "exit",
                            "exit_reason": exit_reason,
                            "flip_blocked": bool(getattr(open_trade, "flip_blocked", False)),
                        }
                    )
                    closed_trade_history.append(exit_result.history_record)
                    open_trade = None
                    next_stop_scan_idx = exec_idx

                if desired != 0 and shares > 0 and take_new_position and open_trade is None and allow_open:
                    size_mult = 1.0
                    overlay_enabled = True
                    regime_spread = np.nan
                    lookback_n = 0

                    eff_shares = shares
                    if allocation_mode == "soft_size" and sizer is not None:
                        side_sizer = long_sizer if desired > 0 else short_sizer
                        risk_signal = Signal(
                            timestamp=ts,
                            symbol=str(row.get("symbol", "SPY")),
                            desired_side=int(desired),
                            confidence=None if np.isnan(p_good) else float(p_good),
                        )
                        eff_shares = side_sizer.size(
                            risk_signal,
                            account={"equity": aum_prev},
                            market_state={
                                "base_qty": shares,
                                "p_good": None if np.isnan(p_good) else float(p_good),
                                "timestamp": ts,
                                "closed_trade_history": closed_trade_history,
                                "row": row,
                            },
                        )
                        details = side_sizer.last_details
                        size_mult = float(details.get("size_mult", 1.0))
                        overlay_enabled = bool(details.get("overlay_enabled", True))
                        regime_spread = details.get("regime_spread", np.nan)
                        lookback_n = int(details.get("lookback_n", 0))
                    elif allocation_mode == "robust_soft_size":
                        overlay_enabled = bool(current_overlay_enabled)
                        if not np.isnan(p_rank):
                            m_raw = size_map_with_neutral_zone(
                                p_rank,
                                used_floor,
                                used_cap,
                                neutral_lo,
                                neutral_hi,
                            )
                            size_mult = float(np.clip(shrink_toward_one(m_raw, shrink_lam), used_floor, used_cap))
                        if not overlay_enabled:
                            size_mult = 1.0
                        eff_shares = _scale_shares(shares, size_mult, aum_prev, lev_cap, exec_px_raw)
                    elif allocation_mode == "convex_rank_size":
                        size_mult, bucket_label = convex_rank_bucket_map(
                            p_rank,
                            low_cut=low_cut,
                            high_cut=high_cut,
                            low_mult=low_mult,
                            mid_mult=mid_mult,
                            high_mult=high_mult,
                            convex_cap_mult=convex_cap_mult,
                        )
                        eff_shares = _scale_shares(shares, size_mult, aum_prev, lev_cap, exec_px_raw)
                    if fast_alpha_overlay:
                        fast_alpha_mult, fast_alpha_favorable = fast_alpha_tactical_multiplier(
                            desired,
                            float(row.get("fast_alpha_ret", np.nan)),
                            favorable_mult=fast_alpha_favorable_mult,
                            unfavorable_mult=fast_alpha_unfavorable_mult,
                        )
                        size_mult, eff_shares = _apply_size_multiplier(
                            size_mult,
                            fast_alpha_mult,
                            shares,
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )

                    if strategy_vol_overlay:
                        size_mult, eff_shares = _apply_size_multiplier(
                            size_mult,
                            strategy_vol_mult,
                            shares,
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )
                    if market_vol_overlay:
                        size_mult, eff_shares = _apply_size_multiplier(
                            size_mult,
                            market_vol_mult,
                            shares,
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )
                    if panic_derisk_overlay:
                        size_mult, eff_shares = _apply_size_multiplier(
                            size_mult,
                            panic_mult,
                            shares,
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )
                    if trend_state_overlay:
                        size_mult, eff_shares = _apply_size_multiplier(
                            size_mult,
                            trend_state_mult,
                            shares,
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )

                    if intraday_risk_overlay:
                        prior_intraday_risk = _history_window_values(
                            risk_history,
                            day,
                            symbol,
                            "intraday_risk_value",
                            intraday_risk_lookback_days,
                        )
                        intraday_risk_mult, intraday_risk_threshold = intraday_risk_size_multiplier(
                            float(row.get("realized_vol_30m", np.nan)),
                            prior_intraday_risk,
                            risk_quantile=intraday_risk_quantile,
                            high_risk_mult=intraday_risk_downsize_mult,
                        )
                        size_mult, eff_shares = _apply_size_multiplier(
                            size_mult,
                            intraday_risk_mult,
                            shares,
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )

                    if execution_chase_control:
                        if execution_chase_relative:
                            (
                                execution_chase_mult_applied,
                                entry_adverse_return,
                                entry_adverse_threshold,
                            ) = execution_aware_relative_entry_multiplier(
                                desired,
                                decision_px,
                                exec_px_raw,
                                band_width_pct=float(row.get("band_width", np.nan)),
                                realized_vol_30m=float(row.get("realized_vol_30m", np.nan)),
                                max_adverse_band_frac=execution_chase_band_frac,
                                max_adverse_vol_mult=execution_chase_vol_mult,
                                adverse_mult=execution_chase_mult,
                            )
                            entry_adverse_bps = float(entry_adverse_return * 10000.0)
                        else:
                            execution_chase_mult_applied, entry_adverse_bps = execution_aware_entry_multiplier(
                                desired,
                                decision_px,
                                exec_px_raw,
                                max_adverse_bps=execution_chase_bps,
                                adverse_mult=execution_chase_mult,
                            )
                        size_mult, eff_shares = _apply_size_multiplier(
                            size_mult,
                            execution_chase_mult_applied,
                            shares,
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )

                    entry_px = apply_execution_spread(exec_px_raw, desired, spread_bps)
                    entry_cost = cost_per_share * eff_shares
                    day_pnl -= entry_cost
                    day_costs += entry_cost
                    total_notional_traded += eff_shares * entry_px
                    size_multipliers.append(size_mult)

                    open_trade = _build_open_trade(
                        desired=desired,
                        shares=eff_shares,
                        entry_timestamp=exec_ts,
                        entry_price=entry_px,
                        entry_cost=entry_cost,
                        decision_timestamp=ts,
                        p_good=p_good,
                        p_rank=p_rank,
                        risk_value=risk_value,
                        used_floor=used_floor,
                        used_cap=used_cap,
                        size_mult=size_mult,
                        bucket_label=bucket_label,
                        overlay_enabled=overlay_enabled,
                        regime_spread=regime_spread,
                        lookback_n=lookback_n,
                        strategy_vol_mult=float(strategy_vol_mult),
                        market_vol_mult=float(market_vol_mult),
                        panic_mult=float(panic_mult),
                        trend_state_mult=float(trend_state_mult),
                        fast_alpha_mult=float(fast_alpha_mult),
                        fast_alpha_favorable=bool(fast_alpha_favorable),
                        intraday_risk_mult=float(intraday_risk_mult),
                        intraday_risk_threshold=float(intraday_risk_threshold),
                        execution_chase_mult=float(execution_chase_mult_applied),
                        entry_adverse_bps=float(entry_adverse_bps),
                        entry_adverse_return=float(entry_adverse_return),
                        entry_adverse_threshold=float(entry_adverse_threshold),
                        base_shares=int(shares),
                    )
                    overlay_enabled_flags.append(overlay_enabled)
                    bucket_labels.append(bucket_label)
                    next_stop_scan_idx = exec_idx
                    if execution_chase_control:
                        execution_chase_history.append(
                            {
                                "timestamp": ts,
                                "date": day,
                                "execution_chase_mult": float(execution_chase_mult_applied),
                                "entry_adverse_bps": float(entry_adverse_bps),
                                "entry_adverse_return": float(entry_adverse_return),
                                "entry_adverse_threshold": float(entry_adverse_threshold),
                            }
                        )
                    if intraday_risk_overlay:
                        intraday_risk_overlay_history.append(
                            {
                                "timestamp": ts,
                                "date": day,
                                "intraday_risk_mult": float(intraday_risk_mult),
                                "intraday_risk_value": float(row.get("realized_vol_30m", np.nan)),
                                "intraday_risk_threshold": float(intraday_risk_threshold),
                            }
                        )
                    if fast_alpha_overlay:
                        fast_alpha_history.append(
                            {
                                "timestamp": ts,
                                "date": day,
                                "fast_alpha_mult": float(fast_alpha_mult),
                                "fast_alpha_ret": float(row.get("fast_alpha_ret", np.nan)),
                                "fast_alpha_favorable": bool(fast_alpha_favorable),
                            }
                        )

                elif (
                    open_trade is not None
                    and desired == current
                    and desired != 0
                    and bool(trend_scalein_enabled)
                    and trend_signal_still_valid(row, current)
                ):
                    open_trade.trend_valid_count = int(getattr(open_trade, "trend_valid_count", 0)) + 1
                    can_boost_today = (not day_boost_used) or (not bool(trend_scalein_once))
                    if (
                        can_boost_today
                        and int(getattr(open_trade, "trend_valid_count", 0)) >= int(trend_persistence_steps)
                    ):
                        exec_idx, exec_row, exec_price_field = get_execution_row(
                            day_df,
                            decision_idx,
                            use_next_bar_open=use_next_bar_open,
                        )
                        exec_px_raw = float(exec_row[exec_price_field])
                        exec_ts = exec_row["timestamp"]
                        target_shares = compute_scalein_target_shares(
                            int(getattr(open_trade, "base_shares", shares)),
                            float(getattr(open_trade, "size_mult", 1.0)),
                            float(trend_boost_mult),
                            float(trend_boost_cap_mult),
                            aum_prev,
                            lev_cap,
                            exec_px_raw,
                        )
                        if open_trade.shares < target_shares:
                            delta_shares = int(target_shares - open_trade.shares)
                            add_px = apply_execution_spread(exec_px_raw, open_trade.side, spread_bps)
                            add_cost = cost_per_share * delta_shares
                            day_pnl -= add_cost
                            day_costs += add_cost
                            total_notional_traded += delta_shares * add_px
                            old_qty = int(open_trade.shares)
                            new_qty = old_qty + delta_shares
                            new_avg = ((open_trade.entry_price * old_qty) + (add_px * delta_shares)) / new_qty
                            open_trade.shares = new_qty
                            open_trade.entry_price = new_avg
                            open_trade.entry_cost = float(open_trade.entry_cost + add_cost)
                            setattr(open_trade, "boost_triggered_day", True)
                            setattr(open_trade, "scale_in_count", int(getattr(open_trade, "scale_in_count", 0)) + 1)
                            setattr(open_trade, "scale_in_shares", int(getattr(open_trade, "scale_in_shares", 0)) + delta_shares)
                            getattr(open_trade, "action_log", ["entry"]).append("scale_in")
                            scale_in_legs.append(
                                {
                                    "timestamp": exec_ts,
                                    "date": day,
                                    "side": "long" if open_trade.side > 0 else "short",
                                    "delta_shares": delta_shares,
                                    "price": add_px,
                                    "costs": add_cost,
                                    "action_type": "scale_in",
                                    "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
                                    "p_rank": getattr(open_trade, "p_rank", np.nan),
                                    "size_mult": getattr(open_trade, "size_mult", np.nan),
                                    "bucket_label": getattr(open_trade, "bucket_label", "mid"),
                                    "overlay_enabled": getattr(open_trade, "overlay_enabled", np.nan),
                                    "strategy_vol_mult": getattr(open_trade, "strategy_vol_mult", np.nan),
                                    "market_vol_mult": getattr(open_trade, "market_vol_mult", np.nan),
                                    "panic_mult": getattr(open_trade, "panic_mult", np.nan),
                                    "trend_state_mult": getattr(open_trade, "trend_state_mult", np.nan),
                                    "fast_alpha_mult": getattr(open_trade, "fast_alpha_mult", np.nan),
                                }
                            )
                            next_stop_scan_idx = exec_idx
                            if bool(trend_scalein_once):
                                day_boost_used = True
                elif open_trade is not None and minute_stop_monitoring:
                    next_stop_scan_idx = max(next_stop_scan_idx, decision_idx + 1)

                if is_candidate:
                    decisions_log.append(
                        {
                            "timestamp": ts,
                            "date": day,
                            "desired": desired,
                            "prev_side": current,
                            "p_good": p_good,
                            "p_rank": p_rank,
                            "threshold": threshold,
                            "accepted": bool(take_new_position),
                            "gate_applied": gate_applied,
                            "flip_reject_mode": flip_reject_mode,
                            "filter_mode": filter_mode,
                            "allocation_mode": allocation_mode,
                            "entry_gate_allowed": bool(allow_open) if desired != 0 else True,
                            "flip_blocked": bool(flip_blocked),
                            "size_mult": getattr(open_trade, "size_mult", size_mult) if open_trade is not None else size_mult,
                            "bucket_label": getattr(open_trade, "bucket_label", bucket_label) if open_trade is not None else bucket_label,
                            "overlay_enabled": getattr(open_trade, "overlay_enabled", overlay_enabled) if open_trade is not None else overlay_enabled,
                            "regime_spread": getattr(open_trade, "regime_spread", regime_spread) if open_trade is not None else regime_spread,
                            "lookback_n": getattr(open_trade, "lookback_n", lookback_n) if open_trade is not None else lookback_n,
                            "strategy_vol_mult": getattr(open_trade, "strategy_vol_mult", strategy_vol_mult) if open_trade is not None else strategy_vol_mult,
                            "market_vol_mult": getattr(open_trade, "market_vol_mult", market_vol_mult) if open_trade is not None else market_vol_mult,
                            "panic_mult": getattr(open_trade, "panic_mult", panic_mult) if open_trade is not None else panic_mult,
                            "trend_state_mult": getattr(open_trade, "trend_state_mult", trend_state_mult) if open_trade is not None else trend_state_mult,
                            "fast_alpha_mult": getattr(open_trade, "fast_alpha_mult", fast_alpha_mult) if open_trade is not None else fast_alpha_mult,
                            "intraday_risk_mult": getattr(open_trade, "intraday_risk_mult", intraday_risk_mult) if open_trade is not None else intraday_risk_mult,
                            "intraday_risk_threshold": getattr(open_trade, "intraday_risk_threshold", intraday_risk_threshold) if open_trade is not None else intraday_risk_threshold,
                            "execution_chase_mult": getattr(open_trade, "execution_chase_mult", execution_chase_mult_applied) if open_trade is not None else execution_chase_mult_applied,
                            "entry_adverse_bps": getattr(open_trade, "entry_adverse_bps", entry_adverse_bps) if open_trade is not None else entry_adverse_bps,
                            "entry_adverse_return": getattr(open_trade, "entry_adverse_return", entry_adverse_return) if open_trade is not None else entry_adverse_return,
                            "entry_adverse_threshold": getattr(open_trade, "entry_adverse_threshold", entry_adverse_threshold) if open_trade is not None else entry_adverse_threshold,
                            "risk_value": risk_value,
                            "used_floor": used_floor,
                            "used_cap": used_cap,
                        }
                    )
                    if not np.isnan(p_good):
                        candidate_score_history.append({"timestamp": ts, "date": day, "symbol": symbol, "p_good": float(p_good)})
                    if not pd.isna(risk_value):
                        risk_history.append(
                            {
                                "timestamp": ts,
                                "date": day,
                                "symbol": symbol,
                                "risk_value": float(risk_value),
                                "intraday_risk_value": float(row.get("realized_vol_30m", np.nan)),
                            }
                        )

            if minute_stop_monitoring:
                stop_scan_start_idx = next_stop_scan_idx

        # Force close any open position at 16:00.
        close_1600 = day_df.loc[day_df["time"] == "16:00"]
        close_row = close_1600.iloc[-1] if not close_1600.empty else day_df.iloc[-1]

        if open_trade is not None and minute_stop_monitoring:
            stop_hit = _first_stop_hit(
                day_df,
                open_trade,
                start_idx=stop_scan_start_idx,
                end_idx=(len(day_df) - 2) if hybrid_stop_mode else (len(day_df) - 1),
                minute_aware=True,
                stop_mode="hybrid" if hybrid_stop_mode else "minute_full",
                catastrophic_stop_bps=catastrophic_stop_bps,
            )
            if stop_hit is not None:
                if hybrid_stop_mode:
                    catastrophic_stop_events += 1
                _, stop_row, raw_exit_px = stop_hit
                ts = stop_row["timestamp"]
                exit_px = apply_execution_spread(raw_exit_px, -open_trade.side, spread_bps)
                exit_result = _close_trade_result(
                    open_trade,
                    exit_timestamp=ts,
                    exit_price=exit_px,
                    cost_per_share=cost_per_share,
                )
                day_pnl += exit_result.pnl_delta
                day_costs += exit_result.costs_delta
                total_notional_traded += exit_result.notional_delta
                trades.append(
                    exit_result.trade_record
                    | {
                        "fast_alpha_mult": getattr(open_trade, "fast_alpha_mult", np.nan),
                        "action_type": "exit",
                        "exit_reason": "stop",
                        "flip_blocked": bool(getattr(open_trade, "flip_blocked", False)),
                    }
                )
                closed_trade_history.append(exit_result.history_record)
                open_trade = None

        if open_trade is not None:
            px = apply_execution_spread(float(close_row["close"]), -open_trade.side, spread_bps)
            ts = close_row["timestamp"]
            exit_result = _close_trade_result(
                open_trade,
                exit_timestamp=ts,
                exit_price=px,
                cost_per_share=cost_per_share,
            )
            day_pnl += exit_result.pnl_delta
            day_costs += exit_result.costs_delta
            total_notional_traded += exit_result.notional_delta
            trades.append(
                exit_result.trade_record
                | {
                    "fast_alpha_mult": getattr(open_trade, "fast_alpha_mult", np.nan),
                    "action_type": "exit",
                    "exit_reason": "eod",
                    "flip_blocked": bool(getattr(open_trade, "flip_blocked", False)),
                }
            )
            closed_trade_history.append(exit_result.history_record)
            open_trade = None

        aum_end = aum_prev + day_pnl
        ret = day_pnl / aum_prev if aum_prev != 0 else 0.0
        base_ret_for_overlay = ret / float(strategy_vol_mult) if strategy_vol_overlay and strategy_vol_mult > 0 else ret
        strategy_base_return_history.append(float(base_ret_for_overlay))
        if strategy_vol_overlay:
            strategy_vol_overlay_history.append(
                {
                    "date": day,
                    "strategy_vol_mult": float(strategy_vol_mult),
                    "target_strategy_vol": float(strategy_target_vol) if not pd.isna(strategy_target_vol) else float("nan"),
                    "current_strategy_vol": float(strategy_current_vol) if not pd.isna(strategy_current_vol) else float("nan"),
                    "base_daily_return": float(base_ret_for_overlay),
                    "overlay_daily_return": float(ret),
                }
            )
        day_close = float(day_df.iloc[-1]["close"])
        base_ret_for_market = float("nan")
        if prev_day_close_for_market is not None and prev_day_close_for_market > 0:
            base_ret_for_market = float(day_close / prev_day_close_for_market - 1.0)
            market_return_history.append(base_ret_for_market)
        prev_day_close_for_market = day_close
        if market_vol_overlay:
            market_vol_overlay_history.append(
                {
                    "date": day,
                    "market_vol_mult": float(market_vol_mult),
                    "target_market_vol": float(market_target_vol) if not pd.isna(market_target_vol) else float("nan"),
                    "current_market_vol": float(market_current_vol) if not pd.isna(market_current_vol) else float("nan"),
                    "base_market_return": float(base_ret_for_market) if not pd.isna(base_ret_for_market) else float("nan"),
                    "overlay_daily_return": float(ret),
                }
            )
        if panic_derisk_overlay:
            panic_overlay_history.append(
                {
                    "date": day,
                    "panic_mult": float(panic_mult),
                    "panic_ret": float(panic_ret) if not pd.isna(panic_ret) else float("nan"),
                    "panic_current_vol": float(panic_current_vol) if not pd.isna(panic_current_vol) else float("nan"),
                    "panic_threshold": float(panic_threshold) if not pd.isna(panic_threshold) else float("nan"),
                    "overlay_daily_return": float(ret),
                }
            )
        if trend_state_overlay:
            trend_state_overlay_history.append(
                {
                    "date": day,
                    "trend_state_mult": float(trend_state_mult),
                    "trend_state_ret": float(trend_state_ret) if not pd.isna(trend_state_ret) else float("nan"),
                    "trend_state_threshold": float(trend_state_threshold) if not pd.isna(trend_state_threshold) else float("nan"),
                    "overlay_daily_return": float(ret),
                }
            )
        close_history.append(day_close)

        equity_rows.append(
            {
                "date": day,
                "equity": aum_end,
                "daily_pnl": day_pnl,
                "daily_return": ret,
                "leverage": leverage,
                "shares": shares,
                "sigma_spy": sigma_spy,
                "costs": day_costs,
            }
        )

        aum_prev = aum_end

    equity_curve = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    decisions_df = pd.DataFrame(decisions_log)
    scale_in_legs_df = pd.DataFrame(scale_in_legs)

    # Final reporting stays separate from the trading loop so tests can compare
    # runtime behavior without depending on log formatting or DataFrame order.
    ml_summary = _summary_with_extras(equity_curve, trades_df, total_notional_traded, initial_aum)
    bucket_series = decisions_df["bucket_label"].dropna() if not decisions_df.empty and "bucket_label" in decisions_df.columns else pd.Series(dtype=object)
    ml_summary.update(
        {
            "accepted_candidates": int(decisions_df["accepted"].sum()) if not decisions_df.empty else 0,
            "candidate_events": int(len(decisions_df)),
            "accept_rate": float(decisions_df["accepted"].mean()) if not decisions_df.empty else 0.0,
            "threshold": float(threshold),
            "flip_reject_mode": flip_reject_mode,
            "filter_mode": filter_mode,
            "allocation_mode": allocation_mode,
            "size_floor": float(size_floor),
            "size_cap": float(size_cap),
            "prob_q20": float(prob_q20) if prob_q20 is not None else float("nan"),
            "prob_q40": float(prob_q40) if prob_q40 is not None else float("nan"),
            "prob_q60": float(prob_q60) if prob_q60 is not None else float("nan"),
            "prob_q80": float(prob_q80) if prob_q80 is not None else float("nan"),
            "avg_size_mult": float(np.mean(size_multipliers)) if size_multipliers else 1.0,
            "scale_in_events": int(len(scale_in_legs_df)),
            "avg_scale_in_size": float(scale_in_legs_df["delta_shares"].mean()) if not scale_in_legs_df.empty else 0.0,
            "pct_low_bucket": float((bucket_series == "low").mean()) if not bucket_series.empty else 0.0,
            "pct_mid_bucket": float((bucket_series == "mid").mean()) if not bucket_series.empty else 0.0,
            "pct_high_bucket": float((bucket_series == "high").mean()) if not bucket_series.empty else 0.0,
            "neutral_zone": bool(neutral_zone),
            "regime_overlay": bool(regime_overlay),
            "regime_lookback_months": int(regime_lookback_months),
            "regime_min_trades": int(regime_min_trades),
            "overlay_enabled_rate": float(np.mean(overlay_enabled_flags)) if overlay_enabled_flags else 0.0,
            "fraction_overlay_enabled": float(np.mean(overlay_enabled_flags)) if overlay_enabled_flags else 0.0,
            "rank_window_days": int(rank_window_days),
            "neutral_lo": float(neutral_lo),
            "neutral_hi": float(neutral_hi),
            "shrink_lam": float(shrink_lam),
            "overlay_gate_lookback_days": int(overlay_gate_lookback_days),
            "risk_quantile": float(risk_quantile),
            "high_risk_floor": float(high_risk_floor),
            "high_risk_cap": float(high_risk_cap),
            "low_cut": float(low_cut),
            "high_cut": float(high_cut),
            "low_mult": float(low_mult),
            "mid_mult": float(mid_mult),
            "high_mult": float(high_mult),
            "convex_cap_mult": float(convex_cap_mult),
            "margin_min_bps": float(margin_min_bps),
            "flip_hysteresis_bps": float(flip_hysteresis_bps),
            "cooldown_steps": int(cooldown_steps),
            "cooldown_on_stop": bool(cooldown_on_stop),
            "trend_scalein_enabled": bool(trend_scalein_enabled),
            "trend_persistence_steps": int(trend_persistence_steps),
            "trend_boost_mult": float(trend_boost_mult),
            "trend_boost_cap_mult": float(trend_boost_cap_mult),
            "trend_scalein_once": bool(trend_scalein_once),
            "strategy_vol_overlay": bool(strategy_vol_overlay),
            "strategy_vol_lookback_days": int(strategy_vol_lookback_days),
            "strategy_vol_floor": float(strategy_vol_floor),
            "strategy_vol_cap": float(strategy_vol_cap),
            "avg_strategy_vol_mult": float(np.mean([x["strategy_vol_mult"] for x in strategy_vol_overlay_history])) if strategy_vol_overlay_history else 1.0,
            "market_vol_overlay": bool(market_vol_overlay),
            "market_vol_lookback_days": int(market_vol_lookback_days),
            "market_vol_floor": float(market_vol_floor),
            "market_vol_cap": float(market_vol_cap),
            "avg_market_vol_mult": float(np.mean([x["market_vol_mult"] for x in market_vol_overlay_history])) if market_vol_overlay_history else 1.0,
            "panic_derisk_overlay": bool(panic_derisk_overlay),
            "panic_return_lookback_days": int(panic_return_lookback_days),
            "panic_vol_lookback_days": int(panic_vol_lookback_days),
            "panic_vol_quantile": float(panic_vol_quantile),
            "panic_exposure": float(panic_exposure),
            "avg_panic_mult": float(np.mean([x["panic_mult"] for x in panic_overlay_history])) if panic_overlay_history else 1.0,
            "trend_state_overlay": bool(trend_state_overlay),
            "trend_state_lookback_days": int(trend_state_lookback_days),
            "trend_state_low_exposure": float(trend_state_low_exposure),
            "trend_state_high_exposure": float(trend_state_high_exposure),
            "avg_trend_state_mult": float(np.mean([x["trend_state_mult"] for x in trend_state_overlay_history])) if trend_state_overlay_history else 1.0,
            "fast_alpha_overlay": bool(fast_alpha_overlay),
            "fast_alpha_bar_mins": int(fast_alpha_bar_mins),
            "fast_alpha_favorable_mult": float(fast_alpha_favorable_mult),
            "fast_alpha_unfavorable_mult": float(fast_alpha_unfavorable_mult),
            "avg_fast_alpha_mult": float(np.mean([x["fast_alpha_mult"] for x in fast_alpha_history])) if fast_alpha_history else 1.0,
            "execution_chase_control": bool(execution_chase_control),
            "execution_chase_relative": bool(execution_chase_relative),
            "execution_chase_bps": float(execution_chase_bps),
            "execution_chase_mult": float(execution_chase_mult),
            "execution_chase_band_frac": float(execution_chase_band_frac),
            "execution_chase_vol_mult": float(execution_chase_vol_mult),
            "avg_execution_chase_mult": float(np.mean([x["execution_chase_mult"] for x in execution_chase_history])) if execution_chase_history else 1.0,
            "avg_entry_adverse_bps": float(np.mean([x["entry_adverse_bps"] for x in execution_chase_history])) if execution_chase_history else 0.0,
            "avg_entry_adverse_return": float(np.mean([x["entry_adverse_return"] for x in execution_chase_history])) if execution_chase_history else 0.0,
            "avg_entry_adverse_threshold": float(np.mean([x["entry_adverse_threshold"] for x in execution_chase_history])) if execution_chase_history else 0.0,
            "hybrid_stop_mode": bool(hybrid_stop_mode),
            "catastrophic_stop_bps": float(catastrophic_stop_bps),
            "catastrophic_stop_events": int(catastrophic_stop_events),
            "intraday_risk_overlay": bool(intraday_risk_overlay),
            "intraday_risk_lookback_days": int(intraday_risk_lookback_days),
            "intraday_risk_quantile": float(intraday_risk_quantile),
            "intraday_risk_downsize_mult": float(intraday_risk_downsize_mult),
            "avg_intraday_risk_mult": float(np.mean([x["intraday_risk_mult"] for x in intraday_risk_overlay_history])) if intraday_risk_overlay_history else 1.0,
        }
    )

    baseline = run_baseline_backtest(
        df,
        initial_aum=initial_aum,
        sigma_target=sigma_target,
        lev_cap=lev_cap,
        commission_per_share=commission_per_share,
        slippage_per_share=slippage_per_share,
        decision_freq_mins=decision_freq_mins,
        first_trade_time=first_trade_time,
        margin_min_bps=margin_min_bps,
        flip_hysteresis_bps=flip_hysteresis_bps,
        cooldown_steps=cooldown_steps,
        cooldown_on_stop=cooldown_on_stop,
        trend_scalein_enabled=trend_scalein_enabled,
        trend_persistence_steps=trend_persistence_steps,
        trend_boost_mult=trend_boost_mult,
        trend_boost_cap_mult=trend_boost_cap_mult,
        trend_scalein_once=trend_scalein_once,
        use_next_bar_open=use_next_bar_open,
        minute_stop_monitoring=minute_stop_monitoring,
        spread_bps=spread_bps,
        daily_sizing=daily_sizing,
        trade_start_date=trade_start_date,
    )
    baseline_summary = baseline["summary"]

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
            "ml_filter": {k: float(ml_summary.get(k, np.nan)) for k in keys},
        }
    )
    comparison["delta"] = comparison["ml_filter"] - comparison["baseline"]

    if not decisions_df.empty:
        if "p_rank" in decisions_df.columns and decisions_df["p_rank"].notna().any():
            logger.info("p_rank distribution: %s", decisions_df["p_rank"].dropna().describe().to_dict())
        if "size_mult" in decisions_df.columns and decisions_df["size_mult"].notna().any():
            logger.info("size_mult distribution: %s", decisions_df["size_mult"].dropna().describe().to_dict())
        if "overlay_enabled" in decisions_df.columns and decisions_df["overlay_enabled"].notna().any():
            logger.info(
                "overlay enabled fraction: %.4f",
                float(decisions_df["overlay_enabled"].fillna(False).astype(float).mean()),
            )

    overlay_history_df = pd.DataFrame(overlay_enabled_history)
    strategy_vol_overlay_df = pd.DataFrame(strategy_vol_overlay_history)
    market_vol_overlay_df = pd.DataFrame(market_vol_overlay_history)
    panic_overlay_df = pd.DataFrame(panic_overlay_history)
    trend_state_overlay_df = pd.DataFrame(trend_state_overlay_history)
    fast_alpha_overlay_df = pd.DataFrame(fast_alpha_history)
    execution_chase_df = pd.DataFrame(execution_chase_history)
    intraday_risk_overlay_df = pd.DataFrame(intraday_risk_overlay_history)

    if size_multipliers:
        logger.info(
            "ML overlay size_mult summary: mean=%.4f min=%.4f max=%.4f",
            float(np.mean(size_multipliers)),
            float(np.min(size_multipliers)),
            float(np.max(size_multipliers)),
        )
    if not decisions_df.empty and "p_rank" in decisions_df.columns:
        ranks = decisions_df["p_rank"].dropna()
        if not ranks.empty:
            logger.info(
                "ML overlay p_rank summary: mean=%.4f min=%.4f max=%.4f",
                float(ranks.mean()),
                float(ranks.min()),
                float(ranks.max()),
            )
    if overlay_enabled_flags:
        logger.info("ML overlay enabled fraction: %.4f", float(np.mean(overlay_enabled_flags)))
    if strategy_vol_overlay_history:
        mults = pd.Series([x["strategy_vol_mult"] for x in strategy_vol_overlay_history], dtype=float)
        logger.info(
            "Strategy-vol overlay multiplier summary: mean=%.4f min=%.4f max=%.4f",
            float(mults.mean()),
            float(mults.min()),
            float(mults.max()),
        )
    if market_vol_overlay_history:
        mults = pd.Series([x["market_vol_mult"] for x in market_vol_overlay_history], dtype=float)
        logger.info(
            "Market-vol overlay multiplier summary: mean=%.4f min=%.4f max=%.4f",
            float(mults.mean()),
            float(mults.min()),
            float(mults.max()),
        )
    if panic_overlay_history:
        mults = pd.Series([x["panic_mult"] for x in panic_overlay_history], dtype=float)
        logger.info(
            "Panic overlay multiplier summary: mean=%.4f min=%.4f max=%.4f",
            float(mults.mean()),
            float(mults.min()),
            float(mults.max()),
        )
    if trend_state_overlay_history:
        mults = pd.Series([x["trend_state_mult"] for x in trend_state_overlay_history], dtype=float)
        logger.info(
            "Trend-state overlay multiplier summary: mean=%.4f min=%.4f max=%.4f",
            float(mults.mean()),
            float(mults.min()),
            float(mults.max()),
        )
    if fast_alpha_history:
        mults = pd.Series([x["fast_alpha_mult"] for x in fast_alpha_history], dtype=float)
        logger.info(
            "Fast Alpha multiplier summary: mean=%.4f min=%.4f max=%.4f",
            float(mults.mean()),
            float(mults.min()),
            float(mults.max()),
        )
    if execution_chase_history:
        mults = pd.Series([x["execution_chase_mult"] for x in execution_chase_history], dtype=float)
        adverse = pd.Series([x["entry_adverse_bps"] for x in execution_chase_history], dtype=float)
        logger.info(
            "Execution-aware entry summary: mean_mult=%.4f mean_adverse_bps=%.4f",
            float(mults.mean()),
            float(adverse.mean()),
        )
    if intraday_risk_overlay_history:
        mults = pd.Series([x["intraday_risk_mult"] for x in intraday_risk_overlay_history], dtype=float)
        logger.info(
            "Intraday-risk overlay multiplier summary: mean=%.4f min=%.4f max=%.4f",
            float(mults.mean()),
            float(mults.min()),
            float(mults.max()),
        )
    if trend_scalein_enabled and int(test_df_days := feat["date"].nunique()) >= 20 and len(scale_in_legs_df) == 0:
        logger.warning("Trend scale-in is enabled but no scale-in events triggered across %d trading days.", int(test_df_days))

    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": ml_summary,
        "baseline_summary": baseline_summary,
        "comparison": comparison,
        "candidate_decisions": decisions_df,
        "overlay_enabled_history": overlay_history_df,
        "strategy_vol_overlay_history": strategy_vol_overlay_df,
        "market_vol_overlay_history": market_vol_overlay_df,
        "panic_overlay_history": panic_overlay_df,
        "trend_state_overlay_history": trend_state_overlay_df,
        "fast_alpha_overlay_history": fast_alpha_overlay_df,
        "execution_chase_history": execution_chase_df,
        "intraday_risk_overlay_history": intraday_risk_overlay_df,
        "scale_in_legs": scale_in_legs_df,
    }


def run_filtered_backtest(df: pd.DataFrame, signals: pd.Series | None = None) -> pd.Series:
    """Backward-compatible wrapper returning ML-filtered daily returns series."""
    _ = signals
    out = run_ml_filtered_backtest(df)
    eq = out["equity_curve"]
    return eq["daily_return"] if not eq.empty else pd.Series(dtype=float)
