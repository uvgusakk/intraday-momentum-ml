"""Baseline intraday strategy backtest with decision-time execution rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from .metrics import summarize_backtest

NY_TZ = "America/New_York"


@dataclass
class OpenTrade:
    """State for an open intraday position."""

    side: int
    shares: int
    entry_timestamp: pd.Timestamp
    entry_price: float
    entry_cost: float


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


def run_baseline_backtest(
    df: pd.DataFrame,
    initial_aum: float = 100000,
    sigma_target: float = 0.02,
    lev_cap: float = 4.0,
    commission_per_share: float = 0.0035,
    slippage_per_share: float = 0.001,
    decision_freq_mins: int = 30,
    first_trade_time: str = "10:00",
) -> dict:
    """Run baseline intraday backtest and return equity, trades, and summary."""
    bars = _normalize_df(df)
    decision_times = _build_decision_time_set(decision_freq_mins, first_trade_time)
    cost_per_share = commission_per_share + slippage_per_share

    sizing = _compute_daily_sizing_table(bars, initial_aum, sigma_target, lev_cap)
    sizing_idx = sizing.set_index("date")

    equity_rows: list[dict] = []
    trades: list[dict] = []

    aum_prev = float(initial_aum)
    total_notional_traded = 0.0

    for day, day_df in bars.groupby("date", sort=True):
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

            # Trailing stop checks are evaluated at decision times only.
            if open_trade is not None and _stop_triggered(open_trade, row):
                exit_cost = cost_per_share * open_trade.shares
                gross = open_trade.side * open_trade.shares * (px - open_trade.entry_price)
                net_close_leg = gross - exit_cost

                day_pnl += net_close_leg
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
                    }
                )
                open_trade = None

            desired = _desired_direction(row)
            current = 0 if open_trade is None else open_trade.side

            if desired != current:
                if open_trade is not None:
                    # Flip/flat: close current first.
                    exit_cost = cost_per_share * open_trade.shares
                    gross = open_trade.side * open_trade.shares * (px - open_trade.entry_price)
                    net_close_leg = gross - exit_cost

                    day_pnl += net_close_leg
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
                        }
                    )
                    open_trade = None

                if desired != 0 and shares > 0:
                    entry_cost = cost_per_share * shares
                    day_pnl -= entry_cost
                    day_costs += entry_cost
                    total_notional_traded += shares * px

                    open_trade = OpenTrade(
                        side=desired,
                        shares=shares,
                        entry_timestamp=ts,
                        entry_price=px,
                        entry_cost=entry_cost,
                    )

        # Force close any position at 16:00 (fallback to last bar if 16:00 missing).
        close_1600 = day_df.loc[day_df["time"] == "16:00"]
        close_row = close_1600.iloc[-1] if not close_1600.empty else day_df.iloc[-1]

        if open_trade is not None:
            px = float(close_row["close"])
            ts = close_row["timestamp"]

            exit_cost = cost_per_share * open_trade.shares
            gross = open_trade.side * open_trade.shares * (px - open_trade.entry_price)
            net_close_leg = gross - exit_cost

            day_pnl += net_close_leg
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

        if day in sizing_idx.index:
            sizing_idx.loc[day, "aum_prev"] = aum_prev
            sizing_idx.loc[day, "shares"] = shares

        aum_prev = aum_end

    equity_curve = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

    summary = summarize_backtest(equity_curve["daily_return"] if not equity_curve.empty else pd.Series(dtype=float))
    summary.update(
        {
            "final_equity": float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(initial_aum),
            "trades_count": int(len(trades_df)),
            "turnover": float(total_notional_traded / equity_curve["equity"].mean()) if not equity_curve.empty and float(equity_curve["equity"].mean()) != 0 else 0.0,
            "total_costs": float(trades_df["costs"].sum()) if not trades_df.empty else 0.0,
        }
    )

    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "summary": summary,
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
