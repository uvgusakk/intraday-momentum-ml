"""ML overlay sizing rules that scale baseline trades without changing direction.

The sizing layer consumes model confidence (`p_good`) and distribution
quantiles learned during training. It never changes the baseline direction
decision; it only maps confidence into a position-size multiplier and can
optionally disable the overlay when recent ranking performance deteriorates.

References:
[1] Lopez de Prado, *Advances in Financial Machine Learning*.
[2] Kissell, *The Science of Algorithmic Trading and Portfolio Management*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..core.types import Signal


@dataclass
class MLOutputSizerRiskManager:
    """Scale baseline quantity using calibrated ML confidence.

    `p_good` is provided at runtime through `Signal.confidence` or
    `market_state["p_good"]`. Threshold and quantiles are loaded once from the
    training artifacts and reused for each decision.
    """

    threshold: float
    prob_q20: float | None = None
    prob_q40: float | None = None
    prob_q60: float | None = None
    prob_q80: float | None = None
    allocation_mode: str = "soft_size"
    neutral_zone: bool = True
    size_floor: float = 0.5
    size_cap: float = 1.5
    regime_overlay: bool = True
    regime_lookback_months: int = 6
    regime_min_trades: int = 80
    last_details: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.allocation_mode != "soft_size":
            raise ValueError("MLOutputSizerRiskManager only supports allocation_mode='soft_size'.")
        if self.size_floor <= 0 or self.size_cap <= 0 or self.size_floor > self.size_cap:
            raise ValueError("size_floor/size_cap must be positive and satisfy size_floor <= size_cap.")

    @classmethod
    def from_artifacts(
        cls,
        threshold_path: str | Path | None = None,
        *,
        allocation_mode: str = "soft_size",
        neutral_zone: bool = True,
        size_floor: float = 0.5,
        size_cap: float = 1.5,
        regime_overlay: bool = True,
        regime_lookback_months: int = 6,
        regime_min_trades: int = 80,
    ) -> "MLOutputSizerRiskManager":
        """Build the risk manager from saved training artifacts."""
        from ..backtest_ml_filter import _load_artifacts

        _, _, threshold, prob_q20, prob_q40, prob_q60, prob_q80 = _load_artifacts(
            None,
            None,
            threshold_path,
        )
        return cls(
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
        )

    def evaluate_regime_overlay(
        self,
        timestamp: pd.Timestamp | None,
        closed_trade_history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Compute whether the ML overlay should be active for the current regime."""
        if not self.regime_overlay:
            return {"overlay_enabled": True, "regime_spread": np.nan, "lookback_n": 0}

        if timestamp is None:
            return {"overlay_enabled": False, "regime_spread": np.nan, "lookback_n": 0}

        history = closed_trade_history or []
        lb = pd.Timestamp(timestamp) - pd.DateOffset(months=self.regime_lookback_months)
        hist = [
            row
            for row in history
            if row["exit_timestamp"] >= lb and not pd.isna(row.get("entry_p_good", np.nan))
        ]
        lookback_n = len(hist)
        if lookback_n < self.regime_min_trades:
            return {"overlay_enabled": False, "regime_spread": np.nan, "lookback_n": lookback_n}

        h = pd.DataFrame(hist)
        h["decile"] = pd.qcut(h["entry_p_good"], 10, labels=False, duplicates="drop")
        top = h.loc[h["decile"] == h["decile"].max(), "pnl"].mean()
        bot = h.loc[h["decile"] == h["decile"].min(), "pnl"].mean()
        spread = float(top - bot)
        return {
            "overlay_enabled": bool(spread > 0),
            "regime_spread": spread,
            "lookback_n": lookback_n,
        }

    def compute_size_multiplier(
        self,
        p_good: float | None,
        *,
        overlay_state: Mapping[str, Any] | None = None,
    ) -> float:
        """Map probability into a clipped sizing multiplier."""
        if overlay_state is not None and not bool(overlay_state.get("overlay_enabled", True)):
            return 1.0
        if p_good is None or pd.isna(p_good):
            return 1.0
        if self.prob_q20 is None or self.prob_q80 is None or self.prob_q80 <= self.prob_q20:
            return 1.0

        if (
            self.neutral_zone
            and self.prob_q40 is not None
            and self.prob_q60 is not None
            and self.prob_q40 < self.prob_q60
            and self.prob_q40 <= float(p_good) <= self.prob_q60
        ):
            return 1.0

        scaled = (float(p_good) - self.prob_q20) / (self.prob_q80 - self.prob_q20)
        scaled = float(np.clip(scaled, 0.0, 1.0))
        mult = self.size_floor + scaled * (self.size_cap - self.size_floor)
        return float(np.clip(mult, self.size_floor, self.size_cap))

    def size(
        self,
        signal: Signal,
        account: Mapping[str, Any],
        market_state: Mapping[str, Any],
    ) -> int:
        """Return the soft-sized quantity for a baseline direction signal."""
        _ = account
        base_qty = int(market_state.get("base_qty", 0))
        if base_qty <= 0:
            self.last_details = {
                "size_mult": 1.0,
                "overlay_enabled": False,
                "regime_spread": np.nan,
                "lookback_n": 0,
                "threshold": self.threshold,
            }
            return 0

        p_good = market_state.get("p_good", signal.confidence)
        timestamp = market_state.get("timestamp")
        if timestamp is None and "row" in market_state:
            timestamp = market_state["row"]["timestamp"]

        overlay_state = self.evaluate_regime_overlay(
            timestamp=pd.Timestamp(timestamp) if timestamp is not None else None,
            closed_trade_history=market_state.get("closed_trade_history"),
        )
        size_mult = self.compute_size_multiplier(p_good, overlay_state=overlay_state)
        qty = int(math.floor(base_qty * size_mult))
        if qty <= 0:
            qty = 1

        self.last_details = {
            "size_mult": size_mult,
            "overlay_enabled": bool(overlay_state["overlay_enabled"]),
            "regime_spread": overlay_state["regime_spread"],
            "lookback_n": int(overlay_state["lookback_n"]),
            "threshold": self.threshold,
            "p_good": p_good,
        }
        return qty
