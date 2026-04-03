"""Protocols for market data, strategy, execution, and risk layers.

We separate these interfaces so research code can swap implementations without
rewriting the whole pipeline. The same strategy can run against historical data,
paper execution, or a live broker adapter as long as each component satisfies
the protocol.

References:
[1] Lopez de Prado, *Advances in Financial Machine Learning*.
[2] Kissell, *The Science of Algorithmic Trading and Portfolio Management*.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .types import Bar, Order, Position, Signal


@runtime_checkable
class MarketDataProvider(Protocol):
    """Historical or live bar provider."""

    def fetch_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str,
    ) -> Iterable[Bar]:
        ...

    def stream_bars(self, symbol: str, timeframe: str) -> Iterable[Bar]:
        """Optional live stream hook for forward-compatible adapters."""
        ...


@runtime_checkable
class Broker(Protocol):
    """Order routing and account state interface."""

    def submit_order(self, order: Order) -> Order:
        ...

    def get_positions(self) -> list[Position]:
        ...

    def get_account(self) -> Mapping[str, Any]:
        ...

    def cancel_all(self) -> None:
        ...


@runtime_checkable
class Strategy(Protocol):
    """Signal generation contract.

    Implementations may use either bar-by-bar callbacks or decision snapshots.
    """

    def on_bar(self, bar: Bar) -> Signal | None:
        ...

    def on_decision(self, snapshot: Mapping[str, Any]) -> Signal | None:
        ...


@runtime_checkable
class RiskManager(Protocol):
    """Position sizing contract."""

    def size(
        self,
        signal: Signal,
        account: Mapping[str, Any],
        market_state: Mapping[str, Any],
    ) -> int:
        ...
