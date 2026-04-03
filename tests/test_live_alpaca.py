"""Unit tests for the additive Alpaca live/paper adapters."""

from __future__ import annotations

import unittest

import pandas as pd

from src.live_alpaca import (
    DataFeed,
    _build_alpaca_order_request,
    _coerce_live_bar_payload,
    _parse_data_feed,
    build_live_enriched_frame,
    compute_live_strategy_snapshot,
)
from src.core.types import Order, OrderType, Side, TimeInForce


class LiveAlpacaHelpersTests(unittest.TestCase):
    def test_parse_data_feed_accepts_valid_names(self) -> None:
        self.assertEqual(_parse_data_feed("iex"), DataFeed.IEX)
        self.assertEqual(_parse_data_feed("delayed_sip"), DataFeed.DELAYED_SIP)

    def test_coerce_live_bar_payload_normalizes_schema(self) -> None:
        payload = {
            "symbol": "spy",
            "timestamp": "2026-04-03T14:31:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1234,
        }
        row = _coerce_live_bar_payload(payload, symbol_hint="SPY")
        self.assertEqual(row["symbol"], "SPY")
        self.assertEqual(str(row["timestamp"].tzinfo), "America/New_York")
        self.assertEqual(row["close"], 100.5)

    def test_live_enriched_frame_and_snapshot_follow_baseline_logic(self) -> None:
        rows = []
        for day in ("2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03"):
            start = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
            for i in range(391):
                ts = start + pd.Timedelta(minutes=i)
                px = 100.0 + 0.01 * i
                if day == "2026-04-03" and ts.strftime("%H:%M") == "15:59":
                    px = 108.0
                rows.append(
                    {
                        "timestamp": ts,
                        "open": px,
                        "high": px + 0.05,
                        "low": px - 0.05,
                        "close": px,
                        "volume": 1000.0,
                        "symbol": "SPY",
                    }
                )
        enriched = build_live_enriched_frame(pd.DataFrame(rows), symbol="SPY")
        snapshot = compute_live_strategy_snapshot(enriched)
        self.assertEqual(snapshot["symbol"], "SPY")
        self.assertIn(snapshot["signal"], {"long", "short", "flat"})
        self.assertTrue(pd.notna(snapshot["timestamp"]))

    def test_build_alpaca_order_request_maps_market_order(self) -> None:
        order = Order(
            symbol="SPY",
            side=Side.LONG,
            qty=3,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
        request = _build_alpaca_order_request(order)
        self.assertEqual(request.symbol, "SPY")
        self.assertEqual(request.qty, 3)
        self.assertEqual(request.side.value, "buy")


if __name__ == "__main__":
    unittest.main()
