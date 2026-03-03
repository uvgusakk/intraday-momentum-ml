"""Backtesting logic that applies an ML probability filter to baseline entries/flips."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .baseline_strategy import (
    OpenTrade,
    _build_decision_time_set,
    _compute_daily_sizing_table,
    _desired_direction,
    _normalize_df,
    _stop_triggered,
    run_baseline_backtest,
)
from .config import load_config
from .features_ml import _compute_feature_frame
from .metrics import summarize_backtest

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "signed_break_distance",
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


def _build_feature_row(row: pd.Series, side: int) -> pd.DataFrame:
    signed_break = (
        (float(row["close"]) - float(row["UB"])) / float(row["UB"])
        if side > 0
        else (float(row["LB"]) - float(row["close"])) / float(row["LB"])
    )

    values = {
        "signed_break_distance": signed_break,
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
    return pd.DataFrame([values], columns=FEATURE_COLS)


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


def run_ml_filtered_backtest(
    df: pd.DataFrame,
    initial_aum: float = 100000,
    sigma_target: float = 0.02,
    lev_cap: float = 4.0,
    commission_per_share: float = 0.0035,
    slippage_per_share: float = 0.001,
    decision_freq_mins: int = 30,
    first_trade_time: str = "10:00",
    model_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
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
) -> dict:
    """Run baseline backtest with ML probability integration.

    Default behavior is soft sizing (`allocation_mode="soft_size"`), which keeps
    baseline trade directions and scales position size by model confidence.
    """
    if flip_reject_mode not in {"hold", "close_flat"}:
        raise ValueError("flip_reject_mode must be one of: 'hold', 'close_flat'")
    if filter_mode not in {"entry_only", "all_candidates"}:
        raise ValueError("filter_mode must be one of: 'entry_only', 'all_candidates'")
    if allocation_mode not in {"hard_filter", "soft_size"}:
        raise ValueError("allocation_mode must be one of: 'hard_filter', 'soft_size'")
    if size_floor <= 0 or size_cap <= 0 or size_floor > size_cap:
        raise ValueError("size_floor/size_cap must be positive and satisfy size_floor <= size_cap")

    model, calibrator, threshold, file_q20, file_q40, file_q60, file_q80 = _load_artifacts(
        model_path, calibration_path, threshold_path
    )
    prob_q20 = file_q20 if prob_q20 is None else prob_q20
    prob_q40 = file_q40 if prob_q40 is None else prob_q40
    prob_q60 = file_q60 if prob_q60 is None else prob_q60
    prob_q80 = file_q80 if prob_q80 is None else prob_q80
    if allocation_mode == "soft_size" and (prob_q20 is None or prob_q80 is None or prob_q80 <= prob_q20):
        logger.warning(
            "Soft sizing requested but prob_q20/prob_q80 are missing or invalid. "
            "Falling back to neutral size multiplier 1.0."
        )

    bars = _normalize_df(df)
    feat = _compute_feature_frame(bars)

    decision_times = _build_decision_time_set(decision_freq_mins, first_trade_time)
    cost_per_share = commission_per_share + slippage_per_share

    sizing = _compute_daily_sizing_table(bars, initial_aum, sigma_target, lev_cap)
    sizing_idx = sizing.set_index("date")

    equity_rows: list[dict] = []
    trades: list[dict] = []
    decisions_log: list[dict] = []
    size_multipliers: list[float] = []
    overlay_enabled_flags: list[bool] = []
    closed_trade_history: list[dict[str, Any]] = []

    aum_prev = float(initial_aum)
    total_notional_traded = 0.0

    for day, day_df in feat.groupby("date", sort=True):
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        day_open = float(day_df.iloc[0]["open"])

        sigma_spy = float(sizing_idx.loc[day, "sigma_spy"]) if day in sizing_idx.index else float("nan")
        leverage = float(sizing_idx.loc[day, "leverage"]) if day in sizing_idx.index else 1.0
        shares = int(math.floor((aum_prev * leverage) / day_open)) if day_open > 0 else 0

        day_pnl = 0.0
        day_costs = 0.0
        open_trade: OpenTrade | None = None

        decisions = day_df.loc[day_df["timestamp"].dt.time.isin(decision_times)]

        for _, row in decisions.iterrows():
            px = float(row["close"])
            ts = row["timestamp"]

            # Keep trailing stops identical to baseline.
            if open_trade is not None and _stop_triggered(open_trade, row):
                exit_cost = cost_per_share * open_trade.shares
                gross = open_trade.side * open_trade.shares * (px - open_trade.entry_price)
                day_pnl += gross - exit_cost
                day_costs += exit_cost
                total_notional_traded += open_trade.shares * px

                trades.append(
                    {
                        "entry_timestamp": open_trade.entry_timestamp,
                        "exit_timestamp": ts,
                        "side": "long" if open_trade.side > 0 else "short",
                        "shares": open_trade.shares,
                        "entry_price": open_trade.entry_price,
                        "exit_price": px,
                        "pnl": gross - open_trade.entry_cost - exit_cost,
                        "costs": open_trade.entry_cost + exit_cost,
                        "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
                    }
                )
                closed_trade_history.append(
                    {
                        "exit_timestamp": ts,
                        "pnl": gross - open_trade.entry_cost - exit_cost,
                        "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
                    }
                )
                open_trade = None

            desired = _desired_direction(row)
            current = 0 if open_trade is None else open_trade.side

            if desired != current:
                # Candidate OPEN/FLIP event when desired != 0 and side differs.
                is_candidate = desired != 0

                p_good = np.nan
                take_new_position = desired != 0
                gate_applied = False

                if is_candidate:
                    gate_applied = filter_mode == "all_candidates" or current == 0

                    # Always score candidate opens/flips when features are available.
                    # In entry_only mode, this lets soft sizing apply to flips as well.
                    X_row = _build_feature_row(row, side=desired)
                    if not X_row.isna().any(axis=None):
                        scores = _raw_scores(model, X_row)
                        p_good = float(calibrator.predict_proba(scores)[0])

                    if allocation_mode == "hard_filter" and gate_applied:
                        # Hard filter only gates events where gating is enabled.
                        take_new_position = (not np.isnan(p_good)) and (p_good >= threshold)
                    else:
                        # Soft sizing and non-gated hard-filter events keep baseline direction changes.
                        take_new_position = True

                should_close_existing = False
                if open_trade is not None:
                    if desired == 0:
                        # Baseline flat signal always closes existing position.
                        should_close_existing = True
                    elif take_new_position:
                        # Accepted flip: close current then open opposite side.
                        should_close_existing = True
                    elif flip_reject_mode == "close_flat":
                        # Rejected flip can optionally close and go flat.
                        should_close_existing = True
                    else:
                        # Rejected flip with hold mode: keep current position.
                        should_close_existing = False

                if should_close_existing and open_trade is not None:
                    exit_cost = cost_per_share * open_trade.shares
                    gross = open_trade.side * open_trade.shares * (px - open_trade.entry_price)
                    day_pnl += gross - exit_cost
                    day_costs += exit_cost
                    total_notional_traded += open_trade.shares * px

                    trades.append(
                        {
                            "entry_timestamp": open_trade.entry_timestamp,
                            "exit_timestamp": ts,
                            "side": "long" if open_trade.side > 0 else "short",
                            "shares": open_trade.shares,
                            "entry_price": open_trade.entry_price,
                            "exit_price": px,
                            "pnl": gross - open_trade.entry_cost - exit_cost,
                            "costs": open_trade.entry_cost + exit_cost,
                            "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
                        }
                    )
                    closed_trade_history.append(
                        {
                            "exit_timestamp": ts,
                            "pnl": gross - open_trade.entry_cost - exit_cost,
                            "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
                        }
                    )
                    open_trade = None

                if desired != 0 and shares > 0 and take_new_position and open_trade is None:
                    size_mult = 1.0
                    overlay_enabled = True
                    regime_spread = np.nan
                    lookback_n = 0

                    if allocation_mode == "soft_size" and regime_overlay:
                        lb = ts - pd.DateOffset(months=regime_lookback_months)
                        hist = [
                            r
                            for r in closed_trade_history
                            if r["exit_timestamp"] >= lb and not pd.isna(r["entry_p_good"])
                        ]
                        lookback_n = len(hist)
                        if lookback_n < regime_min_trades:
                            overlay_enabled = False
                        else:
                            h = pd.DataFrame(hist)
                            h["decile"] = pd.qcut(h["entry_p_good"], 10, labels=False, duplicates="drop")
                            top = h.loc[h["decile"] == h["decile"].max(), "pnl"].mean()
                            bot = h.loc[h["decile"] == h["decile"].min(), "pnl"].mean()
                            regime_spread = float(top - bot)
                            overlay_enabled = regime_spread > 0

                    if allocation_mode == "soft_size" and overlay_enabled and not np.isnan(p_good) and prob_q20 is not None and prob_q80 is not None and prob_q80 > prob_q20:
                        if neutral_zone and prob_q40 is not None and prob_q60 is not None and prob_q40 < prob_q60 and prob_q40 <= p_good <= prob_q60:
                            size_mult = 1.0
                        else:
                            m = (p_good - prob_q20) / (prob_q80 - prob_q20)
                            m = float(np.clip(m, 0.0, 1.0))
                            size_mult = size_floor + m * (size_cap - size_floor)

                    eff_shares = int(math.floor(shares * size_mult))
                    if eff_shares <= 0:
                        eff_shares = 1

                    entry_cost = cost_per_share * eff_shares
                    day_pnl -= entry_cost
                    day_costs += entry_cost
                    total_notional_traded += eff_shares * px
                    size_multipliers.append(size_mult)

                    new_trade = OpenTrade(
                        side=desired,
                        shares=eff_shares,
                        entry_timestamp=ts,
                        entry_price=px,
                        entry_cost=entry_cost,
                    )
                    setattr(new_trade, "entry_p_good", p_good)
                    setattr(new_trade, "size_mult", size_mult)
                    setattr(new_trade, "overlay_enabled", overlay_enabled)
                    setattr(new_trade, "regime_spread", regime_spread)
                    setattr(new_trade, "lookback_n", lookback_n)
                    open_trade = new_trade
                    overlay_enabled_flags.append(overlay_enabled)

                if is_candidate:
                    decisions_log.append(
                        {
                            "timestamp": ts,
                            "date": day,
                            "desired": desired,
                            "prev_side": current,
                            "p_good": p_good,
                            "threshold": threshold,
                            "accepted": bool(take_new_position),
                            "gate_applied": gate_applied,
                            "flip_reject_mode": flip_reject_mode,
                            "filter_mode": filter_mode,
                            "allocation_mode": allocation_mode,
                            "size_mult": getattr(open_trade, "size_mult", np.nan) if open_trade is not None else np.nan,
                            "overlay_enabled": getattr(open_trade, "overlay_enabled", np.nan) if open_trade is not None else np.nan,
                            "regime_spread": getattr(open_trade, "regime_spread", np.nan) if open_trade is not None else np.nan,
                            "lookback_n": getattr(open_trade, "lookback_n", np.nan) if open_trade is not None else np.nan,
                        }
                    )

        # Force close any open position at 16:00.
        close_1600 = day_df.loc[day_df["time"] == "16:00"]
        close_row = close_1600.iloc[-1] if not close_1600.empty else day_df.iloc[-1]

        if open_trade is not None:
            px = float(close_row["close"])
            ts = close_row["timestamp"]

            exit_cost = cost_per_share * open_trade.shares
            gross = open_trade.side * open_trade.shares * (px - open_trade.entry_price)
            day_pnl += gross - exit_cost
            day_costs += exit_cost
            total_notional_traded += open_trade.shares * px

            trades.append(
                {
                    "entry_timestamp": open_trade.entry_timestamp,
                    "exit_timestamp": ts,
                    "side": "long" if open_trade.side > 0 else "short",
                    "shares": open_trade.shares,
                    "entry_price": open_trade.entry_price,
                    "exit_price": px,
                    "pnl": gross - open_trade.entry_cost - exit_cost,
                    "costs": open_trade.entry_cost + exit_cost,
                    "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
                }
            )
            closed_trade_history.append(
                {
                    "exit_timestamp": ts,
                    "pnl": gross - open_trade.entry_cost - exit_cost,
                    "entry_p_good": getattr(open_trade, "entry_p_good", np.nan),
                }
            )
            open_trade = None

        aum_end = aum_prev + day_pnl
        ret = day_pnl / aum_prev if aum_prev != 0 else 0.0

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

    ml_summary = _summary_with_extras(equity_curve, trades_df, total_notional_traded, initial_aum)
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
            "neutral_zone": bool(neutral_zone),
            "regime_overlay": bool(regime_overlay),
            "regime_lookback_months": int(regime_lookback_months),
            "regime_min_trades": int(regime_min_trades),
            "overlay_enabled_rate": float(np.mean(overlay_enabled_flags)) if overlay_enabled_flags else 0.0,
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

    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": ml_summary,
        "baseline_summary": baseline_summary,
        "comparison": comparison,
        "candidate_decisions": decisions_df,
    }


def run_filtered_backtest(df: pd.DataFrame, signals: pd.Series | None = None) -> pd.Series:
    """Backward-compatible wrapper returning ML-filtered daily returns series."""
    _ = signals
    out = run_ml_filtered_backtest(df)
    eq = out["equity_curve"]
    return eq["daily_return"] if not eq.empty else pd.Series(dtype=float)
