from __future__ import annotations

import unittest

import pandas as pd

from src.ml_overlay_robust import (
    compute_overlay_enabled_flag,
    convex_rank_bucket_map,
    execution_aware_entry_multiplier,
    fast_alpha_tactical_multiplier,
    intraday_risk_size_multiplier,
    market_vol_managed_multiplier,
    panic_derisk_multiplier,
    risk_state_multiplier,
    rolling_rank_percentile,
    strategy_vol_managed_multiplier,
    shrink_toward_one,
    size_map_with_neutral_zone,
    trend_state_multiplier,
)


class RobustOverlayUtilsTests(unittest.TestCase):
    def test_rolling_rank_percentile_is_no_lookahead(self) -> None:
        s = pd.Series([0.1, 0.2, 0.3, 0.4])
        out = rolling_rank_percentile(s, window=2)
        self.assertTrue(pd.isna(out.iloc[0]))
        self.assertTrue(pd.isna(out.iloc[1]))
        self.assertAlmostEqual(float(out.iloc[2]), 1.0)
        self.assertAlmostEqual(float(out.iloc[3]), 1.0)

    def test_size_map_with_neutral_zone(self) -> None:
        self.assertEqual(size_map_with_neutral_zone(0.5, 0.85, 1.15, 0.4, 0.6), 1.0)
        self.assertAlmostEqual(size_map_with_neutral_zone(0.0, 0.85, 1.15, 0.4, 0.6), 0.85)
        self.assertAlmostEqual(size_map_with_neutral_zone(1.0, 0.85, 1.15, 0.4, 0.6), 1.15)

    def test_shrink_toward_one(self) -> None:
        self.assertAlmostEqual(shrink_toward_one(1.2, 0.5), 1.1)

    def test_risk_state_multiplier(self) -> None:
        self.assertEqual(risk_state_multiplier(0.3, 0.2, 0.85, 1.15, 0.95, 1.05), (0.95, 1.05))
        self.assertEqual(risk_state_multiplier(0.1, 0.2, 0.85, 1.15, 0.95, 1.05), (0.85, 1.15))

    def test_convex_rank_bucket_map(self) -> None:
        self.assertEqual(convex_rank_bucket_map(0.2), (0.7, "low"))
        self.assertEqual(convex_rank_bucket_map(0.5), (1.0, "mid"))
        self.assertEqual(convex_rank_bucket_map(0.9), (2.0, "high"))
        self.assertEqual(convex_rank_bucket_map(float("nan")), (1.0, "mid"))

    def test_compute_overlay_enabled_flag(self) -> None:
        df = pd.DataFrame({
            'exit_timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'entry_p_good': [i / 30 for i in range(30)],
            'pnl': [-1.0] * 15 + [1.0] * 15,
        })
        self.assertTrue(compute_overlay_enabled_flag(df, 'entry_p_good', 'pnl', 63))
        df['pnl'] = [1.0] * 15 + [-1.0] * 15
        self.assertFalse(compute_overlay_enabled_flag(df, 'entry_p_good', 'pnl', 63))

    def test_strategy_vol_managed_multiplier(self) -> None:
        prior = pd.Series([0.01] * 25 + [0.03] * 25)
        mult, target_vol, current_vol = strategy_vol_managed_multiplier(prior, lookback_days=20, floor=0.5, cap=1.5)
        self.assertFalse(pd.isna(target_vol))
        self.assertFalse(pd.isna(current_vol))
        self.assertLessEqual(mult, 1.0)
        self.assertGreaterEqual(mult, 0.5)

    def test_strategy_vol_managed_multiplier_insufficient_history(self) -> None:
        prior = pd.Series([0.01, -0.02, 0.03])
        mult, target_vol, current_vol = strategy_vol_managed_multiplier(prior, lookback_days=20, floor=0.5, cap=1.5)
        self.assertEqual(mult, 1.0)
        self.assertTrue(pd.isna(target_vol))
        self.assertTrue(pd.isna(current_vol))

    def test_market_vol_managed_multiplier(self) -> None:
        prior = pd.Series([0.005] * 25 + [0.02] * 25)
        mult, target_vol, current_vol = market_vol_managed_multiplier(prior, lookback_days=20, floor=0.5, cap=1.5)
        self.assertFalse(pd.isna(target_vol))
        self.assertFalse(pd.isna(current_vol))
        self.assertLessEqual(mult, 1.0)
        self.assertGreaterEqual(mult, 0.5)

    def test_market_vol_managed_multiplier_insufficient_history(self) -> None:
        prior = pd.Series([0.01, -0.02, 0.03])
        mult, target_vol, current_vol = market_vol_managed_multiplier(prior, lookback_days=20, floor=0.5, cap=1.5)
        self.assertEqual(mult, 1.0)
        self.assertTrue(pd.isna(target_vol))
        self.assertTrue(pd.isna(current_vol))

    def test_panic_derisk_multiplier(self) -> None:
        closes = pd.Series([100 + i for i in range(40)] + [138, 135, 130, 126, 123])
        rets = closes.pct_change().dropna()
        mult, current_ret, current_vol, threshold = panic_derisk_multiplier(
            closes,
            rets,
            return_lookback_days=20,
            vol_lookback_days=20,
            vol_quantile=0.8,
            panic_exposure=0.5,
        )
        self.assertIn(mult, (0.5, 1.0))
        self.assertFalse(pd.isna(current_ret))
        self.assertFalse(pd.isna(current_vol))
        self.assertFalse(pd.isna(threshold))

    def test_trend_state_multiplier(self) -> None:
        closes = pd.Series([100 + i * 0.5 for i in range(140)])
        mult, current_ret, threshold = trend_state_multiplier(closes, lookback_days=60, low_exposure=0.85, high_exposure=1.15)
        self.assertFalse(pd.isna(current_ret))
        self.assertFalse(pd.isna(threshold))
        self.assertIn(mult, (0.85, 1.15))

    def test_fast_alpha_tactical_multiplier(self) -> None:
        self.assertEqual(fast_alpha_tactical_multiplier(1, -0.002, 1.15, 0.85), (1.15, True))
        self.assertEqual(fast_alpha_tactical_multiplier(-1, 0.002, 1.15, 0.85), (1.15, True))
        self.assertEqual(fast_alpha_tactical_multiplier(1, 0.002, 1.15, 0.85), (0.85, False))
        self.assertEqual(fast_alpha_tactical_multiplier(0, 0.002, 1.15, 0.85), (1.0, False))

    def test_execution_aware_entry_multiplier(self) -> None:
        mult, adverse_bps = execution_aware_entry_multiplier(
            1,
            100.0,
            100.15,
            max_adverse_bps=10.0,
            adverse_mult=0.5,
        )
        self.assertEqual(mult, 0.5)
        self.assertGreaterEqual(adverse_bps, 10.0)

        mult, adverse_bps = execution_aware_entry_multiplier(
            -1,
            100.0,
            99.8,
            max_adverse_bps=30.0,
            adverse_mult=0.5,
        )
        self.assertEqual(mult, 1.0)
        self.assertGreaterEqual(adverse_bps, 0.0)

    def test_intraday_risk_size_multiplier(self) -> None:
        mult, threshold = intraday_risk_size_multiplier(
            0.025,
            [0.01] * 10 + [0.015] * 10 + [0.02] * 10,
            risk_quantile=0.8,
            high_risk_mult=0.75,
            min_history=10,
        )
        self.assertEqual(mult, 0.75)
        self.assertFalse(pd.isna(threshold))

        mult, threshold = intraday_risk_size_multiplier(
            0.01,
            [0.01] * 10 + [0.015] * 10 + [0.02] * 10,
            risk_quantile=0.8,
            high_risk_mult=0.75,
            min_history=10,
        )
        self.assertEqual(mult, 1.0)
        self.assertFalse(pd.isna(threshold))


if __name__ == '__main__':
    unittest.main()
