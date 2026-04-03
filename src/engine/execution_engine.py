"""Execution layer for converting signals into paper fills.

This layer isolates signal interpretation, order creation, and fill generation
from strategy logic. That separation keeps backtests easier to audit and makes
it straightforward to replace the paper fill model with a broker adapter later.

References:
[1] Kissell, *The Science of Algorithmic Trading and Portfolio Management*.
[2] Chan, *Algorithmic Trading*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from ..baseline_strategy import apply_execution_spread
from ..core.errors import ExecutionError
from ..core.interfaces import Broker, RiskManager, Strategy
from ..core.types import Bar, Fill, Order, OrderType, Position, Side, Signal, TimeInForce


@dataclass
class PaperBroker:
    """Minimal in-memory broker state for research and backtests."""

    initial_cash: float
    cash: float | None = None
    _positions: dict[str, Position] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = float(self.initial_cash)

    def submit_order(self, order: Order) -> Order:
        return order

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> Mapping[str, Any]:
        return {"cash": float(self.cash), "equity": float(self.cash)}

    def cancel_all(self) -> None:
        return None

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and position state after a paper fill."""
        self.cash -= fill.side.value * fill.qty * fill.price + fill.fees

        current = self._positions.get(fill.symbol)
        if current is None or current.side == Side.FLAT or current.qty == 0:
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                avg_price=fill.price,
            )
            return

        if current.side == fill.side:
            new_qty = current.qty + fill.qty
            new_avg = ((current.avg_price * current.qty) + (fill.price * fill.qty)) / new_qty
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                side=current.side,
                qty=new_qty,
                avg_price=new_avg,
            )
            return

        if fill.qty < current.qty:
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                side=current.side,
                qty=current.qty - fill.qty,
                avg_price=current.avg_price,
            )
            return

        if fill.qty == current.qty:
            self._positions.pop(fill.symbol, None)
            return

        self._positions[fill.symbol] = Position(
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.qty - current.qty,
            avg_price=fill.price,
        )


class ExecutionEngine:
    """Translate strategy intent into orders and paper fills.

    The engine depends only on strategy, risk, and broker interfaces. That keeps
    the order conversion logic reusable across research backtests and future
    live-paper adapters.
    """

    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        broker: Broker,
        fees_per_share: float = 0.0,
        default_order_type: OrderType = OrderType.MARKET,
        default_tif: TimeInForce = TimeInForce.DAY,
    ) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.broker = broker
        self.fees_per_share = float(fees_per_share)
        self.default_order_type = default_order_type
        self.default_tif = default_tif

    def create_order(
        self,
        signal: Signal,
        account: Mapping[str, Any],
        market_state: Mapping[str, Any],
    ) -> Order | None:
        """Convert a non-flat signal into a sized order."""
        if signal.desired_side == Side.FLAT:
            return None

        qty = int(self.risk_manager.size(signal, account, market_state))
        if qty <= 0:
            return None

        return Order(
            symbol=signal.symbol,
            side=signal.desired_side,
            qty=qty,
            order_type=self.default_order_type,
            time_in_force=self.default_tif,
        )

    def execute_order(
        self,
        order: Order,
        bar: Bar,
        *,
        price_field: str = "close",
        price_override: float | None = None,
        spread_bps: float = 0.0,
    ) -> Fill:
        """Apply the paper fill model to the selected bar field."""
        if order.qty <= 0:
            raise ExecutionError("Order quantity must be positive.")
        if bar.symbol != order.symbol:
            raise ExecutionError("Cannot fill order on a bar for a different symbol.")

        self.broker.submit_order(order)
        raw_price = float(price_override) if price_override is not None else float(getattr(bar, price_field))
        fill_price = apply_execution_spread(raw_price, int(order.side), spread_bps)
        fill = Fill(
            timestamp=bar.timestamp,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fees=float(order.qty * self.fees_per_share),
        )
        if hasattr(self.broker, "apply_fill"):
            self.broker.apply_fill(fill)
        return fill

    def execute_signal(
        self,
        signal: Signal,
        account: Mapping[str, Any],
        market_state: Mapping[str, Any],
        bar: Bar,
        *,
        price_field: str = "close",
        price_override: float | None = None,
        spread_bps: float = 0.0,
    ) -> tuple[Order, Fill] | None:
        """Create and execute an order from a signal."""
        order = self.create_order(signal, account, market_state)
        if order is None:
            return None
        fill = self.execute_order(
            order,
            bar,
            price_field=price_field,
            price_override=price_override,
            spread_bps=spread_bps,
        )
        return order, fill
