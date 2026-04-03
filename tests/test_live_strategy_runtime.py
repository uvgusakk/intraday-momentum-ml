"""Unit tests for the additive live strategy runtime."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.core.types import Position, Side
from src.live_strategy_runtime import (
    LivePaperStrategyRunner,
    compute_live_strategy_board,
    compute_live_strategy_snapshot,
    get_live_variant,
    list_live_variants,
)


class _DummyModel:
    feature_names_in_ = np.array(
        [
            "signed_break_distance",
            "break_strength",
            "band_width",
            "vwap_diff",
            "intraday_return",
            "ret_30m",
            "realized_vol_30m",
            "whipsaw_60m",
            "time_of_day_minutes",
            "tod_sin",
            "tod_cos",
        ]
    )

    def decision_function(self, X):
        return np.full(len(X), 0.75, dtype=float)


class _DummyCalibrator:
    def predict_proba(self, scores):
        scores = np.asarray(scores)
        return np.full(scores.shape[0], 0.8, dtype=float)


def _synthetic_live_enriched() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            "2026-04-01 15:59",
            "2026-04-02 15:59",
            "2026-04-03 09:30",
            "2026-04-03 10:00",
        ]
    ).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "SPY",
            "open": [100.0, 101.0, 102.0, 102.2],
            "high": [100.5, 101.5, 102.5, 103.2],
            "low": [99.5, 100.5, 101.8, 102.0],
            "close": [100.2, 101.2, 102.4, 103.1],
            "volume": [1000, 1100, 1300, 1400],
            "date": ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-03"],
            "time": ["15:59", "15:59", "09:30", "10:00"],
            "UB": [100.4, 101.4, 102.1, 102.6],
            "LB": [99.6, 100.6, 101.5, 101.9],
            "VWAP": [100.0, 101.0, 102.0, 102.4],
            "open_0930": [100.0, 101.0, 102.0, 102.0],
            "move_abs": [0.0, 0.0, 0.0, 0.01],
            "sigma": [0.01, 0.01, 0.01, 0.01],
            "break_strength": [0.1, 0.1, 0.2, 0.6],
            "band_width": [0.008, 0.008, 0.006, 0.006],
            "vwap_diff": [0.002, 0.002, 0.003, 0.007],
            "intraday_return": [0.002, 0.002, 0.004, 0.011],
            "ret_30m": [0.001, 0.001, 0.002, 0.004],
            "realized_vol_30m": [0.004, 0.004, 0.004, 0.005],
            "whipsaw_60m": [0.01, 0.01, 0.01, 0.02],
            "time_of_day_minutes": [389.0, 389.0, 0.0, 30.0],
            "tod_sin": [0.0, 0.0, 0.0, 0.5],
            "tod_cos": [1.0, 1.0, 1.0, 0.866],
        }
    )


class LiveStrategyRuntimeTests(unittest.TestCase):
    def test_live_variant_inventory_contains_baseline_and_top_three(self) -> None:
        self.assertEqual(
            list_live_variants(),
            ["baseline", "soft_hybrid_7_5", "soft_hybrid_10", "soft_hybrid_5"],
        )

    def test_compute_live_snapshot_for_baseline(self) -> None:
        enriched = _synthetic_live_enriched()
        snapshot = compute_live_strategy_snapshot(
            enriched,
            variant=get_live_variant("baseline"),
            account_equity=100000.0,
            symbol="SPY",
        )
        self.assertEqual(snapshot["strategy"], "baseline")
        self.assertEqual(snapshot["signal"], "long")
        self.assertTrue(snapshot["is_decision_time"])
        self.assertGreater(snapshot["base_qty"], 0)
        self.assertEqual(snapshot["target_qty"], snapshot["base_qty"])

    def test_compute_live_strategy_board_for_soft_variants(self) -> None:
        enriched = _synthetic_live_enriched()
        bundle = {
            "model": _DummyModel(),
            "calibrator": _DummyCalibrator(),
            "threshold": 0.5,
            "prob_q20": 0.2,
            "prob_q40": 0.4,
            "prob_q60": 0.6,
            "prob_q80": 0.8,
        }
        board = compute_live_strategy_board(
            enriched,
            account_equity=100000.0,
            symbol="SPY",
            artifact_bundle=bundle,
            positions=[
                Position(symbol="SPY", side=Side.LONG, qty=10, avg_price=102.0),
            ],
        )
        self.assertEqual(set(board["strategy"]), set(list_live_variants()))
        soft_row = board.loc[board["strategy"] == "soft_hybrid_7_5"].iloc[0]
        self.assertAlmostEqual(float(soft_row["p_good"]), 0.8, places=6)
        self.assertGreater(int(soft_row["target_qty"]), 0)
        self.assertTrue(bool(soft_row["actionable_now"]))

    def test_live_runner_marks_seeded_bar_as_warming(self) -> None:
        class DummyMarketData:
            def __init__(self, enriched: pd.DataFrame) -> None:
                self._enriched = enriched
                self.is_running = True

            def enriched_bars(self) -> pd.DataFrame:
                return self._enriched

            def start(self) -> None:
                return None

        class DummyBroker:
            def __init__(self) -> None:
                self.is_streaming = True

            def get_positions(self):
                return []

            def get_account(self):
                return {"equity": 100000.0}

            def start_trade_updates(self):
                return None

            def flatten_symbol(self, symbol):
                raise AssertionError("flatten_symbol should not be called during warmup")

        enriched = _synthetic_live_enriched()
        runner = LivePaperStrategyRunner(
            market_data=DummyMarketData(enriched),
            broker=DummyBroker(),
            variant_name="soft_hybrid_7_5",
            symbol="SPY",
            artifact_bundle={
                "model": _DummyModel(),
                "calibrator": _DummyCalibrator(),
                "threshold": 0.5,
                "prob_q20": 0.2,
                "prob_q40": 0.4,
                "prob_q60": 0.6,
                "prob_q80": 0.8,
            },
            dry_run=True,
            min_live_timestamp=pd.Timestamp(enriched["timestamp"].max()),
        )
        snapshot = runner.step()
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["runtime_action"], "warming_live_stream")
        self.assertFalse(bool(snapshot["actionable_now"]))
        self.assertEqual(snapshot["stream_state"], "warming_live_stream")


if __name__ == "__main__":
    unittest.main()
