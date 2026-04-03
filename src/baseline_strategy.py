"""Baseline intraday strategy backtest with decision-time execution rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import math
import warnings

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"


@dataclass
class OpenTrade:
    """State for an open intraday position."""

    side: int
    shares: int
    entry_timestamp: pd.Timestamp
    entry_price: float
    entry_cost: float
    decision_timestamp: pd.Timestamp | None = None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "close", "UB", "LB", "VWAP", "open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ)
    else:
        ts = ts.dt.tz_convert(NY_TZ)

    out["timestamp"] = ts
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["time"] = out["timestamp"].dt.strftime("%H:%M")
    return out.sort_values("timestamp").reset_index(drop=True)


def _build_decision_time_set(decision_freq_mins: int, first_trade_time: str) -> set[time]:
    start = pd.Timestamp(f"2000-01-01 {first_trade_time}")
    end = pd.Timestamp("2000-01-01 15:30")
    if start > end:
        raise ValueError("first_trade_time must be <= 15:30")

    grid = pd.date_range(start=start, end=end, freq=f"{decision_freq_mins}min")
    return {t.time() for t in grid}


def _compute_daily_sizing_table(
    bars: pd.DataFrame,
    initial_aum: float,
    sigma_target: float,
    lev_cap: float,
) -> pd.DataFrame:
    daily = bars.groupby("date", as_index=False, sort=True).agg(
        open_0930=("open", "first"),
        close_1600=("close", "last"),
    )

    daily["close_ret"] = daily["close_1600"].pct_change()
    daily["sigma_spy"] = daily["close_ret"].shift(1).rolling(14, min_periods=2).std(ddof=0)

    def _lev(sig: float) -> float:
        if pd.isna(sig) or sig <= 0:
            return min(1.0, lev_cap)
        return min(lev_cap, sigma_target / float(sig))

    daily["leverage"] = daily["sigma_spy"].map(_lev)
    daily["aum_prev"] = np.nan
    daily.loc[0, "aum_prev"] = initial_aum
    daily["shares"] = 0
    return daily


def _desired_direction(row: pd.Series) -> int:
    if row["close"] > row["UB"]:
        return 1
    if row["close"] < row["LB"]:
        return -1
    return 0


def _stop_triggered(trade: OpenTrade, row: pd.Series) -> bool:
    if trade.side > 0:
        stop = max(float(row["UB"]), float(row["VWAP"]))
        return float(row["close"]) < stop
    stop = min(float(row["LB"]), float(row["VWAP"]))
    return float(row["close"]) > stop


def apply_execution_spread(raw_price: float, side: int, spread_bps: float = 0.0) -> float:
    """Apply a simple half-spread penalty in the adverse direction."""
    price = float(raw_price)
    if price <= 0 or float(spread_bps) <= 0.0 or int(side) == 0:
        return price
    half_spread = float(spread_bps) / 20000.0
    if int(side) > 0:
        return price * (1.0 + half_spread)
    return price * (1.0 - half_spread)


def stop_trigger_details(
    trade: OpenTrade,
    row: pd.Series,
    *,
    minute_aware: bool = False,
) -> tuple[bool, float]:
    """Return whether the stop triggered and the raw exit price to use.

    With ``minute_aware=False`` the legacy behavior is preserved: a stop is
    evaluated on the bar close only. With ``minute_aware=True`` the stop uses a
    simple minute-bar approximation:

    - long stop: trigger when the bar opens below the stop or trades through it
      via the low;
    - short stop: trigger when the bar opens above the stop or trades through it
      via the high.
    """
    if int(trade.side) > 0:
        stop_level = max(float(row["UB"]), float(row["VWAP"]))
        if not minute_aware:
            return (float(row["close"]) < stop_level, float(row["close"]))
        if float(row.get("open", row["close"])) <= stop_level:
            return True, float(row.get("open", row["close"]))
        if float(row.get("low", row["close"])) <= stop_level:
            return True, stop_level
        return False, float(row["close"])

    stop_level = min(float(row["LB"]), float(row["VWAP"]))
    if not minute_aware:
        return (float(row["close"]) > stop_level, float(row["close"]))
    if float(row.get("open", row["close"])) >= stop_level:
        return True, float(row.get("open", row["close"]))
    if float(row.get("high", row["close"])) >= stop_level:
        return True, stop_level
    return False, float(row["close"])


def get_execution_row(
    day_df: pd.DataFrame,
    decision_idx: int,
    *,
    use_next_bar_open: bool = False,
) -> tuple[int, pd.Series, str]:
    """Return the row used for execution and the price field to read from it."""
    if not bool(use_next_bar_open):
        return int(decision_idx), day_df.iloc[int(decision_idx)], "close"
    exec_idx = min(int(decision_idx) + 1, len(day_df) - 1)
    return exec_idx, day_df.iloc[exec_idx], "open"


def compute_break_strength(row: pd.Series, tiny_eps: float = 1e-12) -> float:
    """Compute dimensionless breakout strength relative to current band width."""
    mid = (float(row["UB"]) + float(row["LB"])) / 2.0
    width = max(float(row["UB"]) - float(row["LB"]), tiny_eps)
    return abs(float(row["close"]) - mid) / width


def compute_breakout_margin(row: pd.Series, desired_side: int) -> float:
    """Compute breakout margin outside the active band for the desired side."""
    if int(desired_side) > 0:
        ub = float(row["UB"])
        return max((float(row["close"]) - ub) / ub, 0.0) if ub != 0 else 0.0
    if int(desired_side) < 0:
        lb = float(row["LB"])
        return max((lb - float(row["close"])) / lb, 0.0) if lb != 0 else 0.0
    return 0.0


def flip_allowed_by_hysteresis(
    row: pd.Series,
    current_side: int,
    desired_side: int,
    flip_hysteresis_bps: float = 0.0,
) -> bool:
    """Return whether an opposite-side flip clears the hysteresis margin."""
    if float(flip_hysteresis_bps) <= 0.0:
        return True

    if int(current_side) == 0 or int(desired_side) == 0 or int(desired_side) != -int(current_side):
        return True

    delta = float(flip_hysteresis_bps) / 10000.0
    close = float(row["close"])
    ub = float(row["UB"])
    lb = float(row["LB"])

    if int(current_side) > 0 and int(desired_side) < 0:
        return close < lb * (1.0 - delta)
    if int(current_side) < 0 and int(desired_side) > 0:
        return close > ub * (1.0 + delta)
    return True


def trend_signal_still_valid(row: pd.Series, side: int) -> bool:
    """Return whether the active position still has same-direction breakout confirmation."""
    if int(side) > 0:
        return float(row["close"]) > float(row["UB"])
    if int(side) < 0:
        return float(row["close"]) < float(row["LB"])
    return False


def compute_scalein_target_shares(
    base_shares: int,
    size_mult: float,
    trend_boost_mult: float,
    trend_boost_cap_mult: float,
    aum_prev: float,
    lev_cap: float,
    price: float,
) -> int:
    """Compute capped target shares for a trend-day scale-in event."""
    if base_shares <= 0 or price <= 0:
        return 0
    effective_boost = min(float(trend_boost_mult), float(trend_boost_cap_mult))
    target = int(math.floor(base_shares * float(size_mult) * effective_boost))
    max_qty = int(math.floor((float(aum_prev) * float(lev_cap)) / float(price))) if lev_cap > 0 else target
    return max(0, min(target, max_qty))


def run_baseline_backtest(
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
    daily_sizing: pd.DataFrame | None = None,
    trade_start_date: str | pd.Timestamp | None = None,
) -> dict:
    """Run the baseline backtest through the modular engine wrapper.

    The legacy function signature stays unchanged so the CLI and existing
    notebook code remain compatible while the implementation moves into the new
    strategy/execution/risk separation.
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

    from .engine.backtest_engine import (
        BacktestConfig,
        BacktestEngine,
        FixedQuantityRiskManager,
    )
    from .strategies import BaselineNoiseAreaStrategy

    engine = BacktestEngine(
        strategy=BaselineNoiseAreaStrategy(
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
        ),
        risk_manager=FixedQuantityRiskManager(),
        config=BacktestConfig(
            initial_aum=initial_aum,
            sigma_target=sigma_target,
            lev_cap=lev_cap,
            commission_per_share=commission_per_share,
            slippage_per_share=slippage_per_share,
            decision_freq_mins=decision_freq_mins,
            first_trade_time=first_trade_time,
            use_next_bar_open=use_next_bar_open,
            minute_stop_monitoring=minute_stop_monitoring,
            spread_bps=spread_bps,
        ),
    )
    result = engine.run(
        df,
        trade_start_date=trade_start_date,
        daily_sizing=daily_sizing,
    )
    return {
        "equity_curve": result.equity_curve,
        "trades": result.trades,
        "summary": result.summary,
    }


def generate_baseline_signals(df: pd.DataFrame) -> pd.Series:
    """Generate baseline direction signal from UB/LB breakouts."""
    work = _normalize_df(df)
    signal = np.where(work["close"] > work["UB"], 1, np.where(work["close"] < work["LB"], -1, 0))
    return pd.Series(signal, index=work.index, name="signal")


def compute_strategy_returns(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """Compute simple bar-to-bar strategy returns from aligned signals."""
    work = _normalize_df(df)
    s = pd.Series(signals).reindex(work.index).fillna(0)
    rets = work["close"].pct_change().fillna(0.0)
    return (s.shift(1).fillna(0) * rets).rename("strategy_return")
