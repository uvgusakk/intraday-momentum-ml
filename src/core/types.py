"""Core trading domain types.

The codebase separates domain types from strategy, execution, and backtest logic
so each layer can evolve independently. This keeps market data, signal
generation, order handling, and portfolio accounting easier to test and reason
about across research and production code paths.

References:
[1] Lopez de Prado, *Advances in Financial Machine Learning*.
[2] Chan, *Algorithmic Trading*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum, IntEnum
from typing import Any

import pandas as pd


class Side(IntEnum):
    """Signed position or signal side."""

    SHORT = -1
    FLAT = 0
    LONG = 1

    @classmethod
    def from_value(cls, value: int | "Side") -> "Side":
        """Coerce integer-like inputs into the canonical side enum."""
        if isinstance(value, cls):
            return value
        return cls(int(value))


class OrderType(str, Enum):
    """Supported order types for research and paper execution."""

    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    """Minimal time-in-force set used by the paper engine."""

    DAY = "day"
    GTC = "gtc"


@dataclass(frozen=True)
class Bar:
    """Normalized bar object used by strategy and execution layers."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    date: date
    time: time


@dataclass(frozen=True)
class Signal:
    """Strategy intent at a decision point."""

    timestamp: pd.Timestamp
    symbol: str
    desired_side: Side = Side.FLAT
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "desired_side", Side.from_value(self.desired_side))


@dataclass(frozen=True)
class Order:
    """Executable order request."""

    symbol: str
    side: Side
    qty: int
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", Side.from_value(self.side))
        object.__setattr__(self, "qty", int(self.qty))
        if self.qty < 0:
            raise ValueError("Order.qty must be non-negative.")


@dataclass(frozen=True)
class Fill:
    """Executed order fill."""

    timestamp: pd.Timestamp
    symbol: str
    side: Side
    qty: int
    price: float
    fees: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", Side.from_value(self.side))
        object.__setattr__(self, "qty", int(self.qty))


@dataclass(frozen=True)
class Position:
    """Open position state."""

    symbol: str
    side: Side
    qty: int
    avg_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", Side.from_value(self.side))
        object.__setattr__(self, "qty", int(self.qty))


@dataclass(frozen=True)
class TradeRecord:
    """Completed trade with realized PnL and optional metadata."""

    entry: Fill
    exit: Fill
    side: Side
    qty: int
    pnl: float
    costs: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", Side.from_value(self.side))
        object.__setattr__(self, "qty", int(self.qty))
