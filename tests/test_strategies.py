"""Unit tests for strategy and risk-manager wrappers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.baseline_strategy import compute_break_strength, compute_breakout_margin, run_baseline_backtest
from src.backtest_ml_filter import _hybrid_stop_trigger_details
from src.core.types import Side, Signal
from src.engine.backtest_engine import BacktestConfig, BacktestEngine, FixedQuantityRiskManager
from src.strategies import BaselineNoiseAreaStrategy, MLOutputSizerRiskManager


def _synthetic_enriched_day() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            "2024-01-03 09:30",
            "2024-01-03 10:00",
            "2024-01-03 10:30",
            "2024-01-03 11:00",
            "2024-01-03 16:00",
        ]
    ).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "SPY",
            "open": [100.0, 100.0, 102.0, 103.0, 98.0],
            "high": [100.5, 102.5, 103.5, 103.0, 98.0],
            "low": [99.5, 99.8, 101.8, 97.5, 96.5],
            "close": [100.0, 102.0, 103.0, 98.0, 97.0],
            "volume": [1000, 1200, 1100, 1300, 900],
            "UB": [101.0, 101.0, 102.0, 101.0, 100.0],
            "LB": [99.0, 99.0, 100.0, 99.0, 98.0],
            "VWAP": [100.0, 100.0, 101.0, 100.0, 99.0],
        }
    )


class BaselineNoiseAreaStrategyTests(unittest.TestCase):
    def test_baseline_strategy_decisions_match_legacy_backtest(self) -> None:
        df = _synthetic_enriched_day()
        strategy = BaselineNoiseAreaStrategy()

        decision_rows = df[df["timestamp"].dt.strftime("%H:%M").isin(["10:00", "10:30", "11:00"])]
        sides = [
            int(strategy.on_decision({"row": row, "symbol": "SPY"}).desired_side)
            for _, row in decision_rows.iterrows()
        ]
        self.assertEqual(sides, [1, 1, -1])

        out = run_baseline_backtest(df)
        trades = out["trades"]

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades["side"].tolist(), ["long", "short"])
        self.assertEqual(
            pd.to_datetime(trades["entry_timestamp"]).dt.strftime("%H:%M").tolist(),
            ["10:00", "11:00"],
        )

    def test_next_bar_open_execution_shifts_entry_timestamp(self) -> None:
        ts = pd.to_datetime(
            [
                "2024-01-03 09:30",
                "2024-01-03 10:00",
                "2024-01-03 10:01",
                "2024-01-03 16:00",
            ]
        ).tz_localize("America/New_York")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": "SPY",
                "open": [100.0, 100.0, 102.2, 103.0],
                "high": [100.5, 102.5, 102.8, 103.0],
                "low": [99.5, 99.8, 101.9, 102.8],
                "close": [100.0, 102.0, 102.4, 103.0],
                "volume": [1000, 1200, 1100, 900],
                "UB": [101.0, 101.0, 101.2, 101.5],
                "LB": [99.0, 99.0, 99.2, 99.5],
                "VWAP": [100.0, 100.2, 100.5, 101.0],
            }
        )
        out = run_baseline_backtest(df, use_next_bar_open=True)
        trades = out["trades"]
        self.assertEqual(len(trades), 1)
        self.assertEqual(pd.to_datetime(trades.iloc[0]["decision_timestamp"]).strftime("%H:%M"), "10:00")
        self.assertEqual(pd.to_datetime(trades.iloc[0]["entry_timestamp"]).strftime("%H:%M"), "10:01")

    def test_minute_stop_monitoring_exits_before_next_decision(self) -> None:
        ts = pd.to_datetime(
            [
                "2024-01-03 09:30",
                "2024-01-03 10:00",
                "2024-01-03 10:01",
                "2024-01-03 10:30",
                "2024-01-03 16:00",
            ]
        ).tz_localize("America/New_York")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": "SPY",
                "open": [100.0, 100.0, 102.0, 100.3, 100.3],
                "high": [100.5, 102.5, 102.3, 100.4, 100.4],
                "low": [99.5, 99.8, 100.4, 100.1, 100.1],
                "close": [100.0, 102.0, 101.0, 100.3, 100.3],
                "volume": [1000, 1200, 1100, 1300, 900],
                "UB": [101.0, 101.0, 100.5, 100.5, 100.5],
                "LB": [99.0, 99.0, 99.0, 99.0, 99.0],
                "VWAP": [100.0, 100.0, 100.2, 100.3, 100.3],
            }
        )
        out = run_baseline_backtest(
            df,
            use_next_bar_open=True,
            minute_stop_monitoring=True,
        )
        trades = out["trades"]
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "stop")
        self.assertEqual(pd.to_datetime(trades.iloc[0]["entry_timestamp"]).strftime("%H:%M"), "10:01")
        self.assertEqual(pd.to_datetime(trades.iloc[0]["exit_timestamp"]).strftime("%H:%M"), "10:01")

    def test_hybrid_stop_ignores_marginal_touch_but_triggers_catastrophe(self) -> None:
        trade = type("T", (), {"side": 1})()
        mild_row = pd.Series(
            {
                "open": 101.0,
                "low": 100.96,
                "close": 101.0,
                "UB": 100.9,
                "VWAP": 101.0,
            }
        )
        catastrophic_row = pd.Series(
            {
                "open": 101.0,
                "low": 100.70,
                "close": 100.8,
                "UB": 100.9,
                "VWAP": 101.0,
            }
        )
        stop_hit, _ = _hybrid_stop_trigger_details(trade, mild_row, catastrophic_stop_bps=10.0)
        self.assertFalse(stop_hit)
        stop_hit, raw_exit = _hybrid_stop_trigger_details(trade, catastrophic_row, catastrophic_stop_bps=10.0)
        self.assertTrue(stop_hit)
        self.assertLess(raw_exit, 101.0)


class BreakStrengthTests(unittest.TestCase):
    def test_break_strength_sanity(self) -> None:
        row = pd.Series({"UB": 110.0, "LB": 90.0, "close": 105.0})
        self.assertAlmostEqual(compute_break_strength(row), 0.25)

        row = pd.Series({"UB": 110.0, "LB": 90.0, "close": 100.0})
        self.assertAlmostEqual(compute_break_strength(row), 0.0)

        row = pd.Series({"UB": 110.0, "LB": 90.0, "close": 110.0})
        self.assertAlmostEqual(compute_break_strength(row), 0.5)

        row = pd.Series({"UB": 110.0, "LB": 90.0, "close": 115.0})
        self.assertAlmostEqual(compute_break_strength(row), 0.75)

class BreakoutMarginTests(unittest.TestCase):
    def test_breakout_margin_sanity(self) -> None:
        row = pd.Series({"UB": 100.0, "LB": 90.0, "close": 101.0})
        self.assertAlmostEqual(compute_breakout_margin(row, 1), 0.01)

        row = pd.Series({"UB": 100.0, "LB": 90.0, "close": 89.1})
        self.assertAlmostEqual(compute_breakout_margin(row, -1), 0.01)

        row = pd.Series({"UB": 100.0, "LB": 90.0, "close": 95.0})
        self.assertAlmostEqual(compute_breakout_margin(row, 0), 0.0)



class FlipHysteresisTests(unittest.TestCase):
    def test_flip_hysteresis_blocks_marginal_flip(self) -> None:
        strategy = BaselineNoiseAreaStrategy(flip_hysteresis_bps=10.0)
        ts = pd.Timestamp("2024-01-03 11:00", tz="America/New_York")

        blocked_row = pd.Series({"timestamp": ts, "UB": 102.0, "LB": 100.0, "close": 99.95})
        self.assertFalse(strategy.allow_open(ts, Side.LONG, Side.SHORT, row=blocked_row))

        allowed_row = pd.Series({"timestamp": ts, "UB": 102.0, "LB": 100.0, "close": 99.85})
        self.assertTrue(strategy.allow_open(ts, Side.LONG, Side.SHORT, row=allowed_row))

    def test_blocked_flip_exits_to_flat(self) -> None:
        class NoStopStrategy(BaselineNoiseAreaStrategy):
            def stop_triggered(self, position, row) -> bool:
                return False

        ts = pd.to_datetime(
            [
                "2024-01-03 09:30",
                "2024-01-03 10:00",
                "2024-01-03 10:30",
                "2024-01-03 16:00",
            ]
        ).tz_localize("America/New_York")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": "SPY",
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [100.5, 102.5, 100.2, 100.0],
                "low": [99.5, 99.9, 99.9, 99.8],
                "close": [100.0, 102.0, 99.95, 100.0],
                "volume": [1000, 1200, 1100, 900],
                "UB": [101.0, 101.0, 102.0, 101.0],
                "LB": [99.0, 99.0, 100.0, 99.0],
                "VWAP": [100.0, 100.0, 100.0, 100.0],
            }
        )
        engine = BacktestEngine(
            strategy=NoStopStrategy(flip_hysteresis_bps=10.0),
            risk_manager=FixedQuantityRiskManager(),
            config=BacktestConfig(initial_aum=100000.0, first_trade_time="10:00"),
        )
        out = engine.run(df)
        trades = out.trades
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "flip_blocked_exit")
        self.assertEqual(trades.iloc[0]["side"], "long")
        self.assertTrue(bool(trades.iloc[0]["flip_blocked"]))


class MLOutputSizerRiskManagerTests(unittest.TestCase):
    def test_soft_size_neutral_zone_and_clipping(self) -> None:
        sizer = MLOutputSizerRiskManager(
            threshold=0.5,
            prob_q20=0.2,
            prob_q40=0.4,
            prob_q60=0.6,
            prob_q80=0.8,
            neutral_zone=True,
            size_floor=0.5,
            size_cap=1.5,
            regime_overlay=False,
        )
        ts = pd.Timestamp("2024-01-03 10:00", tz="America/New_York")

        neutral_qty = sizer.size(
            signal=Signal(timestamp=ts, symbol="SPY", desired_side=1, confidence=0.5),
            account={},
            market_state={"base_qty": 100, "p_good": 0.5},
        )
        self.assertEqual(neutral_qty, 100)
        self.assertEqual(sizer.last_details["size_mult"], 1.0)

        low_qty = sizer.size(
            signal=Signal(timestamp=ts, symbol="SPY", desired_side=1, confidence=0.0),
            account={},
            market_state={"base_qty": 100, "p_good": 0.0},
        )
        high_qty = sizer.size(
            signal=Signal(timestamp=ts, symbol="SPY", desired_side=1, confidence=1.0),
            account={},
            market_state={"base_qty": 100, "p_good": 1.0},
        )

        self.assertEqual(low_qty, 50)
        self.assertEqual(high_qty, 150)
        self.assertEqual(sizer.compute_size_multiplier(0.0), 0.5)
        self.assertEqual(sizer.compute_size_multiplier(1.0), 1.5)


class TrendScaleInTests(unittest.TestCase):
    def test_trend_scalein_triggers_once_after_persistence(self) -> None:
        ts = pd.to_datetime(
            [
                "2024-01-03 09:30",
                "2024-01-03 10:00",
                "2024-01-03 10:30",
                "2024-01-03 11:00",
                "2024-01-03 16:00",
            ]
        ).tz_localize("America/New_York")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": "SPY",
                "open": [100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [100.5, 102.5, 103.5, 104.5, 104.0],
                "low": [99.5, 101.5, 102.5, 103.5, 103.0],
                "close": [100.0, 102.0, 103.0, 104.0, 104.0],
                "volume": [1000, 1200, 1100, 1300, 900],
                "UB": [101.0, 101.0, 101.0, 101.0, 101.0],
                "LB": [99.0, 99.0, 99.0, 99.0, 99.0],
                "VWAP": [100.0, 100.5, 101.0, 101.5, 102.0],
            }
        )
        out = run_baseline_backtest(
            df,
            trend_scalein_enabled=True,
            trend_persistence_steps=2,
            trend_boost_mult=1.8,
            trend_boost_cap_mult=2.5,
            trend_scalein_once=True,
        )
        trades = out["trades"]
        self.assertEqual(len(trades), 1)
        self.assertTrue(bool(trades.iloc[0]["boost_triggered_day"]))
        self.assertEqual(int(trades.iloc[0]["scale_in_count"]), 1)
        self.assertEqual(int(trades.iloc[0]["scale_in_shares"]), 800)
        self.assertIn("scale_in", list(trades.iloc[0]["action_log"]))
        self.assertEqual(int(trades.iloc[0]["shares"]), 1800)

    def test_trend_scalein_not_triggered_if_signal_breaks_early(self) -> None:
        ts = pd.to_datetime(
            [
                "2024-01-03 09:30",
                "2024-01-03 10:00",
                "2024-01-03 10:30",
                "2024-01-03 16:00",
            ]
        ).tz_localize("America/New_York")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": "SPY",
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [100.5, 102.5, 100.5, 100.0],
                "low": [99.5, 101.5, 99.5, 99.5],
                "close": [100.0, 102.0, 100.0, 100.0],
                "volume": [1000, 1200, 1100, 900],
                "UB": [101.0, 101.0, 101.0, 101.0],
                "LB": [99.0, 99.0, 99.0, 99.0],
                "VWAP": [100.0, 100.5, 100.2, 100.0],
            }
        )
        out = run_baseline_backtest(
            df,
            trend_scalein_enabled=True,
            trend_persistence_steps=2,
            trend_boost_mult=1.8,
            trend_boost_cap_mult=2.5,
            trend_scalein_once=True,
        )
        trades = out["trades"]
        self.assertEqual(len(trades), 1)
        self.assertFalse(bool(trades.iloc[0]["boost_triggered_day"]))
        self.assertEqual(int(trades.iloc[0]["scale_in_count"]), 0)
        self.assertEqual(int(trades.iloc[0]["scale_in_shares"]), 0)
        self.assertEqual(int(trades.iloc[0]["shares"]), 1000)


if __name__ == "__main__":
    unittest.main()
