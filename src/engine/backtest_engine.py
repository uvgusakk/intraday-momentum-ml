"""Backtest engine that coordinates strategy, risk, and execution.

The engine exists to keep data normalization, decision scheduling, execution,
and portfolio accounting separate from strategy rules. That improves
maintainability for the fellowship assignment because each concern can be
explained, tested, and swapped independently.

References:
[1] Lopez de Prado, *Advances in Financial Machine Learning*.
[2] Chan, *Algorithmic Trading*.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Any, Mapping

import pandas as pd

from ..core.errors import BacktestError, DataValidationError
from ..core.interfaces import RiskManager, Strategy
from ..core.types import Bar, Fill, Order, Position, Side, Signal, TradeRecord
from ..metrics import summarize_backtest
from ..baseline_strategy import (
    compute_scalein_target_shares,
    get_execution_row,
    OpenTrade,
    stop_trigger_details,
)
from ..strategies.baseline_noise_area import BaselineNoiseAreaStrategy
from .execution_engine import ExecutionEngine, PaperBroker

NY_TZ = "America/New_York"


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable backtest configuration for scheduling, fees, and sizing."""

    initial_aum: float = 100000.0
    sigma_target: float = 0.02
    lev_cap: float = 4.0
    commission_per_share: float = 0.0035
    slippage_per_share: float = 0.001
    decision_freq_mins: int = 30
    first_trade_time: str = "10:00"
    use_next_bar_open: bool = False
    minute_stop_monitoring: bool = False
    spread_bps: float = 0.0

    @property
    def fees_per_share(self) -> float:
        return self.commission_per_share + self.slippage_per_share


@dataclass(frozen=True)
class BacktestResult:
    """Structured engine output with dataframes for backward compatibility."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, float]
    raw_trades: list[TradeRecord] = field(default_factory=list)


@dataclass
class ActiveTrade:
    """Open trade state tracked by the engine."""

    position: Position
    entry_fill: Fill
    metadata: dict[str, Any] = field(default_factory=dict)


class BandBreakoutStrategy(BaselineNoiseAreaStrategy):
    """Backward-compatible alias for the baseline noise-area strategy."""


class FixedQuantityRiskManager:
    """Risk manager that delegates quantity to precomputed market state."""

    def size(
        self,
        signal: Signal,
        account: Mapping[str, Any],
        market_state: Mapping[str, Any],
    ) -> int:
        _ = signal, account
        return int(market_state.get("base_qty", 0))


def _normalize_enriched_bars(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "close", "UB", "LB", "VWAP"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ)
    else:
        ts = ts.dt.tz_convert(NY_TZ)

    out["timestamp"] = ts
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["time"] = out["timestamp"].dt.strftime("%H:%M")
    if "symbol" not in out.columns:
        out["symbol"] = "SPY"
    return out.sort_values("timestamp").reset_index(drop=True)


def _row_to_bar(row: pd.Series) -> Bar:
    ts = pd.Timestamp(row["timestamp"])
    return Bar(
        timestamp=ts,
        open=float(row["open"]),
        high=float(row.get("high", row["close"])),
        low=float(row.get("low", row["close"])),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
        symbol=str(row.get("symbol", "SPY")),
        date=ts.date(),
        time=ts.time(),
    )


def _build_decision_time_set(decision_freq_mins: int, first_trade_time: str) -> set[dt_time]:
    start = pd.Timestamp(f"2000-01-01 {first_trade_time}")
    end = pd.Timestamp("2000-01-01 15:30")
    if start > end:
        raise BacktestError("first_trade_time must be <= 15:30")
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
    return daily


def _desired_side_from_row(row: pd.Series) -> Side:
    if float(row["close"]) > float(row["UB"]):
        return Side.LONG
    if float(row["close"]) < float(row["LB"]):
        return Side.SHORT
    return Side.FLAT


def _stop_triggered(position: Position, row: pd.Series) -> bool:
    if position.side == Side.LONG:
        stop = max(float(row["UB"]), float(row["VWAP"]))
        return float(row["close"]) < stop
    stop = min(float(row["LB"]), float(row["VWAP"]))
    return float(row["close"]) > stop


def _trade_record_from_fills(entry: Fill, exit_fill: Fill, metadata: Mapping[str, Any] | None = None) -> TradeRecord:
    gross = entry.side.value * entry.qty * (exit_fill.price - entry.price)
    costs = float(entry.fees + exit_fill.fees)
    return TradeRecord(
        entry=entry,
        exit=exit_fill,
        side=entry.side,
        qty=entry.qty,
        pnl=float(gross - costs),
        costs=costs,
        metadata=dict(metadata or {}),
    )


class BacktestEngine:
    """Drive decision-time backtests over enriched bars.

    The engine owns the scheduling loop, stop logic, execution, and portfolio
    accounting while strategy implementations focus only on signal generation.
    """

    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        config: BacktestConfig | None = None,
        broker: PaperBroker | None = None,
        market_state_builder: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config or BacktestConfig()
        self.broker = broker or PaperBroker(initial_cash=self.config.initial_aum)
        self.market_state_builder = market_state_builder
        self.execution_engine = ExecutionEngine(
            strategy=strategy,
            risk_manager=risk_manager,
            broker=self.broker,
            fees_per_share=self.config.fees_per_share,
        )

    def run(
        self,
        df: pd.DataFrame,
        *,
        trade_start_date: str | pd.Timestamp | None = None,
        daily_sizing: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run the backtest and return equity, trades, and summary."""
        bars = _normalize_enriched_bars(df)
        decision_times = _build_decision_time_set(
            self.config.decision_freq_mins,
            self.config.first_trade_time,
        )
        sizing = (
            daily_sizing.copy()
            if daily_sizing is not None
            else _compute_daily_sizing_table(
                bars,
                self.config.initial_aum,
                self.config.sigma_target,
                self.config.lev_cap,
            )
        )
        if "date" not in sizing.columns:
            raise BacktestError("daily_sizing must contain a 'date' column")
        sizing = sizing.copy()
        sizing["date"] = pd.to_datetime(sizing["date"]).dt.strftime("%Y-%m-%d")
        sizing = sizing.set_index("date")
        live_start = None if trade_start_date is None else pd.Timestamp(trade_start_date).strftime("%Y-%m-%d")

        aum_prev = float(self.config.initial_aum)
        total_notional_traded = 0.0
        equity_rows: list[dict[str, Any]] = []
        trade_records: list[TradeRecord] = []
        closed_trade_history: list[dict[str, Any]] = []

        for day, day_df in bars.groupby("date", sort=True):
            day_df = day_df.sort_values("timestamp").reset_index(drop=True)
            if live_start is not None and str(day) < live_start:
                continue
            cooldown_until_ts: pd.Timestamp | None = None
            day_boost_used = False
            day_open = float(day_df.iloc[0]["open"])
            sigma_spy = float(sizing.loc[day, "sigma_spy"]) if day in sizing.index else float("nan")
            leverage = float(sizing.loc[day, "leverage"]) if day in sizing.index else 1.0
            base_qty = int(math.floor((aum_prev * leverage) / day_open)) if day_open > 0 else 0

            day_pnl = 0.0
            day_costs = 0.0
            active_trade: ActiveTrade | None = None
            stop_scan_start_idx = 0

            decision_indices = day_df.index[day_df["timestamp"].dt.time.isin(decision_times)].tolist()
            for decision_idx in decision_indices:
                row = day_df.iloc[decision_idx]
                bar = _row_to_bar(row)

                next_stop_scan_idx = stop_scan_start_idx
                if active_trade is not None:
                    if self.config.minute_stop_monitoring:
                        for scan_idx in range(stop_scan_start_idx, decision_idx + 1):
                            scan_row = day_df.iloc[scan_idx]
                            stop_hit, raw_stop_price = self._stop_trigger_details(
                                active_trade.position,
                                scan_row,
                                minute_aware=True,
                            )
                            if not stop_hit:
                                continue
                            stop_bar = _row_to_bar(scan_row)
                            exit_fill = self._close_trade(
                                active_trade,
                                stop_bar,
                                price_override=raw_stop_price,
                            )
                            record = self._record_closed_trade(
                                trade_records,
                                closed_trade_history,
                                active_trade,
                                exit_fill,
                                exit_reason="stop",
                            )
                            day_pnl += record.pnl + active_trade.entry_fill.fees
                            day_costs += float(exit_fill.fees)
                            total_notional_traded += exit_fill.qty * exit_fill.price
                            if getattr(self.strategy, "cooldown_on_stop", True) and int(getattr(self.strategy, "cooldown_steps", 0)) > 0:
                                cooldown_until_ts = stop_bar.timestamp + pd.Timedelta(
                                    minutes=self.config.decision_freq_mins * int(getattr(self.strategy, "cooldown_steps", 0))
                                )
                            active_trade = None
                            next_stop_scan_idx = scan_idx + 1
                            break
                        else:
                            next_stop_scan_idx = max(next_stop_scan_idx, decision_idx + 1)
                    elif self._stop_triggered(active_trade.position, row):
                        exit_fill = self._close_trade(active_trade, bar)
                        record = self._record_closed_trade(
                            trade_records,
                            closed_trade_history,
                            active_trade,
                            exit_fill,
                            exit_reason="stop",
                        )
                        day_pnl += record.pnl + active_trade.entry_fill.fees
                        day_costs += float(exit_fill.fees)
                        total_notional_traded += exit_fill.qty * exit_fill.price
                        if getattr(self.strategy, "cooldown_on_stop", True) and int(getattr(self.strategy, "cooldown_steps", 0)) > 0:
                            cooldown_until_ts = bar.timestamp + pd.Timedelta(
                                minutes=self.config.decision_freq_mins * int(getattr(self.strategy, "cooldown_steps", 0))
                            )
                        active_trade = None

                signal = self._generate_signal(row, bar, active_trade)
                desired = signal.desired_side if signal is not None else Side.FLAT
                current = active_trade.position.side if active_trade is not None else Side.FLAT

                if desired != current:
                    allow_open = True
                    flip_blocked = False
                    if desired != Side.FLAT and signal is not None and hasattr(self.strategy, "allow_open"):
                        allow_open = bool(self.strategy.allow_open(bar.timestamp, current, desired, row=row))
                        if current != Side.FLAT and desired == Side(-int(current)) and not allow_open:
                            flip_blocked = True
                    if desired != Side.FLAT and cooldown_until_ts is not None and bar.timestamp < cooldown_until_ts:
                        allow_open = False

                    if flip_blocked and active_trade is not None:
                        active_trade.metadata["flip_blocked"] = True

                    should_close_existing = False
                    blocked_flip_exit = False
                    if active_trade is not None:
                        if desired == Side.FLAT:
                            should_close_existing = True
                        elif allow_open:
                            should_close_existing = True
                        elif flip_blocked:
                            should_close_existing = True
                            blocked_flip_exit = True

                    exec_idx, exec_row, exec_price_field = get_execution_row(
                        day_df,
                        decision_idx,
                        use_next_bar_open=self.config.use_next_bar_open,
                    )
                    exec_bar = _row_to_bar(exec_row)

                    if should_close_existing and active_trade is not None:
                        exit_fill = self._close_trade(
                            active_trade,
                            exec_bar,
                            price_field=exec_price_field,
                        )
                        if desired == Side.FLAT:
                            exit_reason = "flat"
                        elif blocked_flip_exit:
                            exit_reason = "flip_blocked_exit"
                        else:
                            exit_reason = "flip"
                        record = _trade_record_from_fills(
                            active_trade.entry_fill,
                            exit_fill,
                            {**active_trade.metadata, "exit_reason": exit_reason},
                        )
                        trade_records.append(record)
                        day_pnl += record.pnl + active_trade.entry_fill.fees
                        day_costs += float(exit_fill.fees)
                        total_notional_traded += exit_fill.qty * exit_fill.price
                        active_trade = None
                        next_stop_scan_idx = exec_idx

                    if desired != Side.FLAT and base_qty > 0 and signal is not None and allow_open and active_trade is None:
                        account = {"equity": aum_prev, "cash": self.broker.cash}
                        market_state = {
                            "row": row,
                            "base_qty": base_qty,
                            "date": day,
                            "leverage": leverage,
                            "sigma_spy": sigma_spy,
                            "timestamp": bar.timestamp,
                            "closed_trade_history": closed_trade_history,
                        }
                        if self.market_state_builder is not None:
                            extra_state = self.market_state_builder(
                                row=row,
                                bar=bar,
                                signal=signal,
                                active_trade=active_trade,
                                market_state=market_state,
                                closed_trade_history=closed_trade_history,
                                equity=aum_prev,
                                day=day,
                            )
                            if extra_state:
                                market_state.update(dict(extra_state))
                        result = self.execution_engine.execute_signal(
                            signal,
                            account,
                            market_state,
                            exec_bar,
                            price_field=exec_price_field,
                            spread_bps=self.config.spread_bps,
                        )
                        if result is not None:
                            _, entry_fill = result
                            day_pnl -= entry_fill.fees
                            day_costs += float(entry_fill.fees)
                            total_notional_traded += entry_fill.qty * entry_fill.price
                            metadata = {
                                "reason": "decision_entry",
                                "flip_blocked": False,
                                "base_qty": int(base_qty),
                                "trend_valid_count": 0,
                                "boost_triggered_day": False,
                                "scale_in_count": 0,
                                "scale_in_shares": 0,
                                "action_log": ["entry"],
                                "decision_timestamp": bar.timestamp,
                            }
                            p_good = market_state.get("p_good", signal.confidence)
                            if p_good is not None and not pd.isna(p_good):
                                metadata["entry_p_good"] = float(p_good)
                            if hasattr(self.risk_manager, "last_details"):
                                metadata.update(dict(getattr(self.risk_manager, "last_details")))
                            active_trade = ActiveTrade(
                                position=Position(
                                    symbol=entry_fill.symbol,
                                    side=entry_fill.side,
                                    qty=entry_fill.qty,
                                    avg_price=entry_fill.price,
                                ),
                                entry_fill=entry_fill,
                                metadata=metadata,
                            )
                            next_stop_scan_idx = exec_idx

                elif (
                    active_trade is not None
                    and desired == current
                    and desired != Side.FLAT
                    and bool(getattr(self.strategy, "trend_scalein_enabled", False))
                ):
                    trend_valid = True
                    if hasattr(self.strategy, "trend_signal_still_valid"):
                        trend_valid = bool(self.strategy.trend_signal_still_valid(row, current))
                    meta = active_trade.metadata
                    if trend_valid:
                        meta["trend_valid_count"] = int(meta.get("trend_valid_count", 0)) + 1
                    else:
                        meta["trend_valid_count"] = 0
                    can_boost_today = (not day_boost_used) or (not bool(getattr(self.strategy, "trend_scalein_once", True)))
                    if (
                        trend_valid
                        and can_boost_today
                        and int(meta.get("trend_valid_count", 0)) >= int(getattr(self.strategy, "trend_persistence_steps", 2))
                    ):
                        exec_idx, exec_row, exec_price_field = get_execution_row(
                            day_df,
                            decision_idx,
                            use_next_bar_open=self.config.use_next_bar_open,
                        )
                        exec_bar = _row_to_bar(exec_row)
                        reference_price = float(getattr(exec_bar, exec_price_field))
                        target_qty = compute_scalein_target_shares(
                            int(meta.get("base_qty", base_qty)),
                            1.0,
                            float(getattr(self.strategy, "trend_boost_mult", 1.8)),
                            float(getattr(self.strategy, "trend_boost_cap_mult", 2.5)),
                            aum_prev,
                            self.config.lev_cap,
                            reference_price,
                        )
                        if active_trade.position.qty < target_qty:
                            delta_qty = int(target_qty - active_trade.position.qty)
                            add_order = Order(symbol=bar.symbol, side=active_trade.position.side, qty=delta_qty)
                            add_fill = self.execution_engine.execute_order(
                                order=add_order,
                                bar=exec_bar,
                                price_field=exec_price_field,
                                spread_bps=self.config.spread_bps,
                            )
                            day_pnl -= add_fill.fees
                            day_costs += float(add_fill.fees)
                            total_notional_traded += add_fill.qty * add_fill.price
                            old_qty = int(active_trade.position.qty)
                            new_qty = old_qty + add_fill.qty
                            new_avg = ((active_trade.position.avg_price * old_qty) + (add_fill.price * add_fill.qty)) / new_qty
                            active_trade.position = Position(
                                symbol=active_trade.position.symbol,
                                side=active_trade.position.side,
                                qty=new_qty,
                                avg_price=new_avg,
                            )
                            active_trade.entry_fill = Fill(
                                timestamp=active_trade.entry_fill.timestamp,
                                symbol=active_trade.entry_fill.symbol,
                                side=active_trade.entry_fill.side,
                                qty=new_qty,
                                price=new_avg,
                                fees=float(active_trade.entry_fill.fees + add_fill.fees),
                            )
                            meta["boost_triggered_day"] = True
                            meta["scale_in_count"] = int(meta.get("scale_in_count", 0)) + 1
                            meta["scale_in_shares"] = int(meta.get("scale_in_shares", 0)) + int(add_fill.qty)
                            meta.setdefault("action_log", ["entry"]).append("scale_in")
                            if bool(getattr(self.strategy, "trend_scalein_once", True)):
                                day_boost_used = True
                            next_stop_scan_idx = exec_idx

                if self.config.minute_stop_monitoring:
                    stop_scan_start_idx = next_stop_scan_idx

            close_row = day_df.loc[day_df["time"] == "16:00"]
            closing_bar = _row_to_bar(close_row.iloc[-1] if not close_row.empty else day_df.iloc[-1])
            if active_trade is not None and self.config.minute_stop_monitoring:
                for scan_idx in range(stop_scan_start_idx, len(day_df)):
                    scan_row = day_df.iloc[scan_idx]
                    stop_hit, raw_stop_price = self._stop_trigger_details(
                        active_trade.position,
                        scan_row,
                        minute_aware=True,
                    )
                    if not stop_hit:
                        continue
                    stop_bar = _row_to_bar(scan_row)
                    exit_fill = self._close_trade(
                        active_trade,
                        stop_bar,
                        price_override=raw_stop_price,
                    )
                    record = self._record_closed_trade(
                        trade_records,
                        closed_trade_history,
                        active_trade,
                        exit_fill,
                        exit_reason="stop",
                    )
                    day_pnl += record.pnl + active_trade.entry_fill.fees
                    day_costs += float(exit_fill.fees)
                    total_notional_traded += exit_fill.qty * exit_fill.price
                    active_trade = None
                    break
            if active_trade is not None:
                exit_fill = self._close_trade(active_trade, closing_bar)
                record = self._record_closed_trade(
                    trade_records,
                    closed_trade_history,
                    active_trade,
                    exit_fill,
                    exit_reason="eod",
                )
                day_pnl += record.pnl + active_trade.entry_fill.fees
                day_costs += float(exit_fill.fees)
                total_notional_traded += exit_fill.qty * exit_fill.price
                active_trade = None

            aum_end = aum_prev + day_pnl
            daily_return = day_pnl / aum_prev if aum_prev != 0 else 0.0
            equity_rows.append(
                {
                    "date": day,
                    "equity": aum_end,
                    "daily_pnl": day_pnl,
                    "daily_return": daily_return,
                    "leverage": leverage,
                    "shares": base_qty,
                    "sigma_spy": sigma_spy,
                    "costs": day_costs,
                }
            )
            aum_prev = aum_end

        equity_curve = pd.DataFrame(equity_rows)
        trades_df = self._trade_records_to_df(trade_records)
        summary = summarize_backtest(
            equity_curve["daily_return"] if not equity_curve.empty else pd.Series(dtype=float)
        )
        summary.update(
            {
                "final_equity": float(equity_curve["equity"].iloc[-1])
                if not equity_curve.empty
                else float(self.config.initial_aum),
                "trades_count": int(len(trades_df)),
                "turnover": float(total_notional_traded / equity_curve["equity"].mean())
                if not equity_curve.empty and float(equity_curve["equity"].mean()) != 0
                else 0.0,
                "total_costs": float(trades_df["costs"].sum()) if not trades_df.empty else 0.0,
            }
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            summary=summary,
            raw_trades=trade_records,
        )

    def _generate_signal(self, row: pd.Series, bar: Bar, active_trade: ActiveTrade | None) -> Signal | None:
        snapshot = {
            "row": row,
            "bar": bar,
            "symbol": bar.symbol,
            "position": active_trade.position if active_trade is not None else None,
        }
        if hasattr(self.strategy, "on_decision"):
            signal = self.strategy.on_decision(snapshot)
            if signal is not None:
                return signal
        if hasattr(self.strategy, "on_bar"):
            return self.strategy.on_bar(bar)
        raise BacktestError("Strategy must implement on_decision(...) or on_bar(...).")

    def _close_trade(
        self,
        active_trade: ActiveTrade,
        bar: Bar,
        *,
        price_field: str = "close",
        price_override: float | None = None,
    ) -> Fill:
        exit_order_side = Side.SHORT if active_trade.position.side == Side.LONG else Side.LONG
        exit_order = Order(symbol=bar.symbol, side=exit_order_side, qty=active_trade.position.qty)
        return self.execution_engine.execute_order(
            order=exit_order,
            bar=bar,
            price_field=price_field,
            price_override=price_override,
            spread_bps=self.config.spread_bps,
        )

    @staticmethod
    def _record_closed_trade(
        trade_records: list[TradeRecord],
        closed_trade_history: list[dict[str, Any]],
        active_trade: ActiveTrade,
        exit_fill: Fill,
        exit_reason: str,
    ) -> TradeRecord:
        record = _trade_record_from_fills(
            active_trade.entry_fill,
            exit_fill,
            {**active_trade.metadata, "exit_reason": exit_reason},
        )
        trade_records.append(record)
        closed_trade_history.append(
            {
                "exit_timestamp": exit_fill.timestamp,
                "pnl": record.pnl,
                "entry_p_good": record.metadata.get("entry_p_good", float("nan")),
            }
        )
        return record

    def _stop_triggered(self, position: Position, row: pd.Series) -> bool:
        if hasattr(self.strategy, "stop_triggered"):
            return bool(self.strategy.stop_triggered(position, row))
        return _stop_triggered(position, row)

    def _stop_trigger_details(
        self,
        position: Position,
        row: pd.Series,
        *,
        minute_aware: bool,
    ) -> tuple[bool, float]:
        # Reuse the same simple stop-price approximation across baseline and ML.
        return stop_trigger_details(
            OpenTrade(
                side=int(position.side),
                shares=int(position.qty),
                entry_timestamp=pd.Timestamp(row["timestamp"]),
                entry_price=float(position.avg_price),
                entry_cost=0.0,
            ),
            row,
            minute_aware=minute_aware,
        )

    @staticmethod
    def _trade_records_to_df(trade_records: list[TradeRecord]) -> pd.DataFrame:
        rows = [
            {
                "entry_timestamp": record.entry.timestamp,
                "decision_timestamp": record.metadata.get("decision_timestamp", record.entry.timestamp),
                "exit_timestamp": record.exit.timestamp,
                "side": "long" if record.side == Side.LONG else "short",
                "shares": record.qty,
                "entry_price": record.entry.price,
                "exit_price": record.exit.price,
                "pnl": record.pnl,
                "costs": record.costs,
                "entry_p_good": record.metadata.get("entry_p_good", float("nan")),
                "size_mult": record.metadata.get("size_mult", float("nan")),
                "overlay_enabled": record.metadata.get("overlay_enabled", float("nan")),
                "regime_spread": record.metadata.get("regime_spread", float("nan")),
                "lookback_n": record.metadata.get("lookback_n", float("nan")),
                "exit_reason": record.metadata.get("exit_reason", ""),
                "flip_blocked": bool(record.metadata.get("flip_blocked", False)),
                "action_type": "flip" if record.metadata.get("exit_reason", "") == "flip" else "exit",
                "boost_triggered_day": bool(record.metadata.get("boost_triggered_day", False)),
                "scale_in_count": int(record.metadata.get("scale_in_count", 0)),
                "scale_in_shares": int(record.metadata.get("scale_in_shares", 0)),
                "action_log": record.metadata.get("action_log", []),
                "metadata": record.metadata,
            }
            for record in trade_records
        ]
        return pd.DataFrame(rows)
