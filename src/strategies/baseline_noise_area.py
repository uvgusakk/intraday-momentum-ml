"""Baseline noise-area strategy.

This class packages the existing breakout and stop rules behind a strategy
object. The intent is to keep the signal logic independent from execution and
portfolio sizing while still reusing the legacy helper functions that already
define the baseline behavior.

References:
[1] Chan, *Algorithmic Trading*.
[2] Lopez de Prado, *Advances in Financial Machine Learning*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Mapping

import pandas as pd

from ..baseline_strategy import (
    OpenTrade,
    _build_decision_time_set,
    compute_breakout_margin,
    flip_allowed_by_hysteresis,
    trend_signal_still_valid,
    _desired_direction,
    _stop_triggered,
)
from ..core.types import Position, Side, Signal


@dataclass
class BaselineNoiseAreaStrategy:
    """Baseline direction and stop rules using enriched UB/LB/VWAP columns.

    The strategy itself does not place orders. It emits desired directions at
    scheduled decision times and exposes stop logic so the backtest engine can
    keep execution concerns separate from the signal rules.
    """

    symbol: str = "SPY"
    decision_freq_mins: int = 30
    first_trade_time: str = "10:00"
    flatten_time: str = "16:00"
    margin_min_bps: float = 0.0
    flip_hysteresis_bps: float = 0.0
    cooldown_steps: int = 0
    cooldown_on_stop: bool = True
    trend_scalein_enabled: bool = False
    trend_persistence_steps: int = 2
    trend_boost_mult: float = 1.8
    trend_boost_cap_mult: float = 2.5
    trend_scalein_once: bool = True
    _decision_times: set[time] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._decision_times = _build_decision_time_set(
            self.decision_freq_mins,
            self.first_trade_time,
        )

    def on_bar(self, bar: Any) -> Signal | None:
        _ = bar
        return None

    def on_decision(self, snapshot: Mapping[str, Any]) -> Signal:
        row = snapshot["row"]
        ts = pd.Timestamp(row["timestamp"])
        if ts.strftime("%H:%M") >= self.flatten_time:
            desired = Side.FLAT
        else:
            desired = Side(int(_desired_direction(row)))
        return Signal(
            timestamp=ts,
            symbol=str(row.get("symbol", snapshot.get("symbol", self.symbol))),
            desired_side=desired,
        )

    def is_decision_time(self, timestamp: pd.Timestamp) -> bool:
        return pd.Timestamp(timestamp).time() in self._decision_times

    def should_flatten(self, timestamp: pd.Timestamp) -> bool:
        return pd.Timestamp(timestamp).strftime("%H:%M") >= self.flatten_time

    def allow_open(
        self,
        timestamp: pd.Timestamp,
        current_side: Side,
        desired_side: Side,
        row: pd.Series | None = None,
    ) -> bool:
        _ = timestamp
        if desired_side == Side.FLAT:
            return False
        if row is None:
            return True
        if current_side == Side.FLAT:
            margin_min = float(self.margin_min_bps) / 10000.0
            return compute_breakout_margin(row, int(desired_side)) >= margin_min
        if desired_side == Side(-int(current_side)):
            return flip_allowed_by_hysteresis(
                row,
                int(current_side),
                int(desired_side),
                float(self.flip_hysteresis_bps),
            )
        return True

    def stop_triggered(self, position: Position, row: pd.Series) -> bool:
        legacy_trade = OpenTrade(
            side=int(position.side),
            shares=int(position.qty),
            entry_timestamp=pd.Timestamp(row["timestamp"]),
            entry_price=float(position.avg_price),
            entry_cost=0.0,
        )
        return bool(_stop_triggered(legacy_trade, row))

    def trend_signal_still_valid(self, row: pd.Series, current_side: Side) -> bool:
        return trend_signal_still_valid(row, int(current_side))
