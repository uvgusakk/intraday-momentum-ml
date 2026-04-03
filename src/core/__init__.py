"""Core domain layer for typed trading objects and integration interfaces."""

from .errors import BacktestError, CoreError, DataValidationError, ExecutionError, StrategyError
from .interfaces import Broker, MarketDataProvider, RiskManager, Strategy
from .types import Bar, Fill, Order, OrderType, Position, Side, Signal, TimeInForce, TradeRecord

__all__ = [
    "BacktestError",
    "Bar",
    "Broker",
    "CoreError",
    "DataValidationError",
    "ExecutionError",
    "Fill",
    "MarketDataProvider",
    "Order",
    "OrderType",
    "Position",
    "RiskManager",
    "Side",
    "Signal",
    "Strategy",
    "StrategyError",
    "TimeInForce",
    "TradeRecord",
]
