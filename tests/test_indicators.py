"""Regression tests for indicator calculations."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.indicators import compute_gap_adjusted_bands


class GapAdjustedBandsTests(unittest.TestCase):
    def test_gap_adjusted_bands_use_sigma_only(self) -> None:
        ts = pd.to_datetime(
            [
                "2024-01-02 09:30",
                "2024-01-02 10:00",
                "2024-01-02 16:00",
                "2024-01-03 09:30",
                "2024-01-03 10:00",
                "2024-01-03 16:00",
            ]
        ).tz_localize("America/New_York")
        base = pd.DataFrame(
            {
                "timestamp": ts,
                "date": ts.strftime("%Y-%m-%d"),
                "time": ts.strftime("%H:%M"),
                "open": [100.0, 100.0, 100.0, 102.0, 102.0, 102.0],
                "close": [100.0, 101.0, 99.5, 102.0, 103.0, 101.5],
                "sigma": [0.01, 0.01, 0.01, 0.012, 0.012, 0.012],
            }
        )

        out = compute_gap_adjusted_bands(base)

        expected_anchor_upper = np.maximum(102.0, 99.5)
        expected_anchor_lower = np.minimum(102.0, 99.5)
        self.assertAlmostEqual(float(out.loc[3, "UB"]), expected_anchor_upper * (1.0 + 0.012))
        self.assertAlmostEqual(float(out.loc[3, "LB"]), expected_anchor_lower * (1.0 - 0.012))


if __name__ == "__main__":
    unittest.main()
