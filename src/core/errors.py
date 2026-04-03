"""Shared exception types for engine and adapter layers."""

from __future__ import annotations


class CoreError(Exception):
    """Base exception for the modular trading core."""


class DataValidationError(CoreError):
    """Raised when required market data fields are missing or malformed."""


class StrategyError(CoreError):
    """Raised when a strategy emits invalid state or signals."""


class ExecutionError(CoreError):
    """Raised when orders or fills cannot be processed."""


class BacktestError(CoreError):
    """Raised when the backtest engine receives inconsistent inputs."""
