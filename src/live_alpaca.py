"""Live Alpaca adapters for notebook demos and paper trading.

This module is intentionally additive:

- it does not change the backtest or score-forward code paths;
- it reuses the existing preprocessing and indicator pipeline for live charts;
- it defaults to paper trading only and refuses live order submission unless
  explicitly allowed.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.live.stock import StockDataStream
    from alpaca.data.models.bars import Bar as AlpacaBar
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide
    from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.stream import TradingStream

    ALPACA_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised implicitly in environments without alpaca-py
    class DataFeed(Enum):
        IEX = "iex"
        SIP = "sip"
        DELAYED_SIP = "delayed_sip"
        OTC = "otc"
        BOATS = "boats"
        OVERNIGHT = "overnight"

    class AlpacaOrderSide(Enum):
        BUY = "buy"
        SELL = "sell"

    class AlpacaTimeInForce(Enum):
        DAY = "day"
        GTC = "gtc"

    @dataclass
    class MarketOrderRequest:
        symbol: str
        qty: int
        side: AlpacaOrderSide
        time_in_force: AlpacaTimeInForce

    @dataclass
    class LimitOrderRequest(MarketOrderRequest):
        limit_price: float | None = None

    StockDataStream = None
    AlpacaBar = None
    TradingClient = None
    TradingStream = None
    ALPACA_SDK_AVAILABLE = False

from .baseline_strategy import _desired_direction, compute_breakout_margin
from .config import load_config
from .core.interfaces import Broker, MarketDataProvider
from .core.types import Bar, Order, OrderType, Position, Side, TimeInForce
from .indicators import (
    compute_gap_adjusted_bands,
    compute_intraday_move_from_open,
    compute_sigma_profile,
    compute_vwap,
)
from .preprocess import preprocess_bars

NY_TZ = "America/New_York"
BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]


def _require_alpaca_sdk() -> None:
    if not ALPACA_SDK_AVAILABLE:
        raise ImportError(
            "alpaca-py is required for live streaming and order submission. "
            "Install it in the active environment before using src.live_alpaca runtime adapters."
        )


def _stop_stream_adapter(stream: Any) -> None:
    """Stop an Alpaca websocket stream across SDK variants."""
    if stream is None:
        return

    try:
        result = stream.stop_ws()
        if asyncio.iscoroutine(result):
            asyncio.run(result)
        return
    except Exception:
        pass

    try:
        result = stream.stop()
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    except Exception:
        pass


def _fetch_minute_bars_adapter(*, symbol: str, start: str, end: str, adjustment: str = "raw", force: bool = False) -> pd.DataFrame:
    """Import the historical Alpaca fetch helper lazily.

    The live notebook helpers and unit tests should stay importable even when
    `alpaca-py` is missing. Historical fetches are only needed once a user
    actually seeds or requests bars from Alpaca.
    """
    from .data_alpaca import fetch_minute_bars

    return fetch_minute_bars(
        symbol=symbol,
        start=start,
        end=end,
        adjustment=adjustment,
        feed=os.getenv("ALPACA_LIVE_FEED", "iex").strip().lower() or "iex",
        force=force,
    )


def _parse_data_feed(value: str | DataFeed | None) -> DataFeed:
    """Parse a string feed name into the Alpaca SDK enum."""
    if isinstance(value, DataFeed):
        return value
    raw = str(value or os.getenv("ALPACA_LIVE_FEED", "iex")).strip().lower()
    mapping = {feed.value.lower(): feed for feed in DataFeed}
    if raw not in mapping:
        raise ValueError(f"Unsupported Alpaca live feed: {raw}. Allowed: {sorted(mapping)}")
    return mapping[raw]


def _optional_stream_url(env_name: str) -> str | None:
    """Return a websocket override only when the env var is explicitly ws/wss."""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return None
    if raw.startswith("ws://") or raw.startswith("wss://"):
        return raw
    return None


def _coerce_live_bar_payload(payload: Any, *, symbol_hint: str) -> dict[str, Any]:
    """Normalize Alpaca live bar payloads into the project dataframe schema."""
    if AlpacaBar is not None and isinstance(payload, AlpacaBar):
        data = payload.model_dump()
    elif hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        raise TypeError(f"Unsupported bar payload type: {type(payload)!r}")

    ts = pd.Timestamp(data["timestamp"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(NY_TZ)
    symbol = str(data.get("symbol") or symbol_hint).upper()
    return {
        "timestamp": ts,
        "open": float(data["open"]),
        "high": float(data["high"]),
        "low": float(data["low"]),
        "close": float(data["close"]),
        "volume": float(data.get("volume", 0.0) or 0.0),
        "symbol": symbol,
    }


def _normalize_live_frame(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Return a deduplicated live bars dataframe in the canonical schema."""
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY_TZ)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY_TZ)
    if "symbol" not in out.columns:
        out["symbol"] = str(symbol).upper()
    out["symbol"] = out["symbol"].fillna(str(symbol).upper()).astype(str).str.upper()
    out = out[BAR_COLUMNS].drop_duplicates(subset=["timestamp"], keep="last")
    return out.sort_values("timestamp").reset_index(drop=True)


def build_live_enriched_frame(raw_bars: pd.DataFrame, *, symbol: str = "SPY") -> pd.DataFrame:
    """Run the existing bar-prep pipeline on a live/historical bars dataframe."""
    if raw_bars.empty:
        return pd.DataFrame(columns=BAR_COLUMNS + ["date", "time", "open_0930", "move_abs", "sigma", "UB", "LB", "VWAP"])
    base = _normalize_live_frame(raw_bars, symbol=symbol)
    clean = preprocess_bars(base[["timestamp", "open", "high", "low", "close", "volume"]])
    clean["symbol"] = str(symbol).upper()
    enriched = compute_intraday_move_from_open(clean)
    enriched = compute_sigma_profile(enriched, lookback_days=14)
    enriched = compute_gap_adjusted_bands(enriched)
    enriched = compute_vwap(enriched)
    enriched["symbol"] = str(symbol).upper()
    return enriched


def compute_live_strategy_snapshot(enriched_bars: pd.DataFrame) -> dict[str, Any]:
    """Summarize the latest live state using the baseline decision logic."""
    if enriched_bars.empty:
        return {
            "timestamp": pd.NaT,
            "symbol": "",
            "signal": "no_data",
            "desired_side": 0,
            "close": float("nan"),
            "UB": float("nan"),
            "LB": float("nan"),
            "VWAP": float("nan"),
            "breakout_margin_bps": float("nan"),
        }

    latest = enriched_bars.iloc[-1]
    desired = int(_desired_direction(latest))
    signal = {1: "long", -1: "short", 0: "flat"}[desired]
    margin_bps = float(compute_breakout_margin(latest, desired) * 10000.0) if desired != 0 else 0.0
    return {
        "timestamp": pd.Timestamp(latest["timestamp"]),
        "symbol": str(latest.get("symbol", "")),
        "signal": signal,
        "desired_side": desired,
        "close": float(latest["close"]),
        "UB": float(latest["UB"]),
        "LB": float(latest["LB"]),
        "VWAP": float(latest["VWAP"]),
        "breakout_margin_bps": margin_bps,
        "date": str(latest["date"]),
        "time": str(latest["time"]),
        "rows_today": int((enriched_bars["date"] == latest["date"]).sum()),
    }


def render_live_strategy_chart(
    enriched_bars: pd.DataFrame,
    *,
    symbol: str = "SPY",
    title_suffix: str = "",
) -> tuple[plt.Figure, tuple[Any, Any]]:
    """Render the current-day price, bands, VWAP, and volume for notebook use."""
    if enriched_bars.empty:
        raise ValueError("Cannot render live chart from an empty dataframe")

    latest_day = enriched_bars["date"].iloc[-1]
    day_df = enriched_bars.loc[enriched_bars["date"] == latest_day].copy()
    snapshot = compute_live_strategy_snapshot(enriched_bars)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    ax_price, ax_vol = axes
    ax_price.plot(day_df["timestamp"], day_df["close"], label="Close", linewidth=2.0, color="#1f77b4")
    ax_price.plot(day_df["timestamp"], day_df["UB"], label="UB", linewidth=1.4, linestyle="--", color="#d62728")
    ax_price.plot(day_df["timestamp"], day_df["LB"], label="LB", linewidth=1.4, linestyle="--", color="#2ca02c")
    ax_price.plot(day_df["timestamp"], day_df["VWAP"], label="VWAP", linewidth=1.4, color="#9467bd")
    ax_price.scatter(
        [snapshot["timestamp"]],
        [snapshot["close"]],
        color="#ff7f0e",
        s=70,
        zorder=5,
        label=f"Latest ({snapshot['signal']})",
    )
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="best")
    ax_price.set_title(
        f"{symbol.upper()} Live Strategy View {title_suffix}".strip()
        + f"\n{snapshot['timestamp']} | signal={snapshot['signal']} | breakout_margin_bps={snapshot['breakout_margin_bps']:.2f}"
    )

    ax_vol.bar(day_df["timestamp"], day_df["volume"], color="#888888", width=0.0009)
    ax_vol.set_ylabel("Volume")
    ax_vol.set_xlabel("Time")

    fig.tight_layout()
    return fig, (ax_price, ax_vol)


class AlpacaLiveMarketData(MarketDataProvider):
    """Background live market-data adapter with a thread-safe rolling bar buffer."""

    def __init__(
        self,
        *,
        symbol: str = "SPY",
        feed: str | DataFeed | None = None,
        raw_data: bool = False,
        history_business_days: int = 20,
    ) -> None:
        _require_alpaca_sdk()
        self.config = load_config()
        self.symbol = str(symbol).upper()
        self.feed = _parse_data_feed(feed)
        self.raw_data = bool(raw_data)
        self.history_business_days = int(history_business_days)
        self._lock = threading.Lock()
        self._bars = pd.DataFrame(columns=BAR_COLUMNS)
        self._stream: StockDataStream | None = None
        self._thread: threading.Thread | None = None

    def fetch_bars(self, symbol: str, start: str, end: str, timeframe: str = "1Min") -> list[Bar]:
        """Protocol-compatible historical fetch using the existing Alpaca helper."""
        if str(timeframe).lower() not in {"1min", "1m", "minute"}:
            raise ValueError("AlpacaLiveMarketData.fetch_bars currently supports only 1-minute bars.")
        df = _fetch_minute_bars_adapter(symbol=symbol, start=start, end=end, adjustment="raw", force=False)
        out: list[Bar] = []
        for row in df.itertuples(index=False):
            ts = pd.Timestamp(row.timestamp)
            out.append(
                Bar(
                    timestamp=ts,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    symbol=str(symbol).upper(),
                    date=ts.date(),
                    time=ts.time(),
                )
            )
        return out

    def stream_bars(self, symbol: str, timeframe: str = "1Min") -> list[Bar]:
        """Expose the current rolling buffer through the MarketDataProvider protocol."""
        _ = timeframe
        df = self.bars_df()
        return self.fetch_bars(symbol, str(df["timestamp"].min()), str(df["timestamp"].max()), timeframe) if not df.empty else []

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def seed_history(
        self,
        *,
        lookback_business_days: int | None = None,
        end: pd.Timestamp | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Warm the live buffer with recent historical bars so indicators are defined."""
        end_ts = pd.Timestamp.now(tz=NY_TZ).floor("min") if end is None else pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(NY_TZ)
        else:
            end_ts = end_ts.tz_convert(NY_TZ)
        business_days = int(lookback_business_days or self.history_business_days)
        start_ts = (end_ts - pd.tseries.offsets.BDay(business_days)).floor("D")
        df = _fetch_minute_bars_adapter(
            symbol=self.symbol,
            start=start_ts.strftime("%Y-%m-%d %H:%M"),
            end=end_ts.strftime("%Y-%m-%d %H:%M"),
            adjustment="raw",
            force=force,
        )
        df["symbol"] = self.symbol
        with self._lock:
            self._bars = _normalize_live_frame(df, symbol=self.symbol)
            return self._bars.copy()

    async def _on_bar(self, payload: Any) -> None:
        row = _coerce_live_bar_payload(payload, symbol_hint=self.symbol)
        with self._lock:
            updated = pd.concat([self._bars, pd.DataFrame([row])], ignore_index=True)
            self._bars = _normalize_live_frame(updated, symbol=self.symbol)

    def start(self, *, subscribe_updated_bars: bool = True) -> None:
        """Start the Alpaca websocket stream in a background thread."""
        if self.is_running:
            return
        self._stream = StockDataStream(
            self.config.alpaca_api_key,
            self.config.alpaca_api_secret,
            raw_data=self.raw_data,
            feed=self.feed,
            url_override=_optional_stream_url("ALPACA_STREAM_URL"),
        )
        self._stream.subscribe_bars(self._on_bar, self.symbol)
        if subscribe_updated_bars:
            self._stream.subscribe_updated_bars(self._on_bar, self.symbol)

        def _runner() -> None:
            assert self._stream is not None
            self._stream.run()

        self._thread = threading.Thread(target=_runner, name=f"alpaca-live-{self.symbol}", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop the background websocket stream."""
        if self._stream is not None:
            _stop_stream_adapter(self._stream)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._stream = None
        self._thread = None

    def bars_df(self) -> pd.DataFrame:
        """Return a thread-safe copy of the rolling raw bar buffer."""
        with self._lock:
            return self._bars.copy()

    def enriched_bars(self) -> pd.DataFrame:
        """Return the live buffer with the standard preprocessing and indicators applied."""
        return build_live_enriched_frame(self.bars_df(), symbol=self.symbol)


def _alpaca_order_side(side: Side) -> AlpacaOrderSide:
    if Side.from_value(side) == Side.LONG:
        return AlpacaOrderSide.BUY
    if Side.from_value(side) == Side.SHORT:
        return AlpacaOrderSide.SELL
    raise ValueError("Cannot submit an order with Side.FLAT")


def _alpaca_tif(value: TimeInForce) -> AlpacaTimeInForce:
    mapping = {
        TimeInForce.DAY: AlpacaTimeInForce.DAY,
        TimeInForce.GTC: AlpacaTimeInForce.GTC,
    }
    return mapping[value]


def _build_alpaca_order_request(order: Order) -> MarketOrderRequest | LimitOrderRequest:
    """Translate the project order type into the Alpaca SDK request model."""
    common = {
        "symbol": order.symbol.upper(),
        "qty": int(order.qty),
        "side": _alpaca_order_side(order.side),
        "time_in_force": _alpaca_tif(order.time_in_force),
    }
    if order.order_type == OrderType.MARKET:
        return MarketOrderRequest(**common)
    if order.order_type == OrderType.LIMIT:
        if order.limit_price is None:
            raise ValueError("Limit orders require limit_price")
        return LimitOrderRequest(limit_price=float(order.limit_price), **common)
    raise ValueError(f"Unsupported order type: {order.order_type}")


class AlpacaPaperBroker(Broker):
    """Paper-trading broker wrapper around Alpaca's Trading API."""

    def __init__(
        self,
        *,
        raw_data: bool = False,
        allow_live: bool = False,
    ) -> None:
        _require_alpaca_sdk()
        self.config = load_config()
        self.paper = "paper" in self.config.alpaca_base_url.lower()
        if not self.paper and not allow_live:
            raise ValueError(
                "Refusing non-paper trading because ALPACA_BASE_URL is not a paper endpoint. "
                "Pass allow_live=True only if you explicitly want live order routing."
            )
        self._client = TradingClient(
            self.config.alpaca_api_key,
            self.config.alpaca_api_secret,
            paper=self.paper,
            raw_data=raw_data,
            url_override=self.config.alpaca_base_url,
        )
        self._stream: TradingStream | None = None
        self._thread: threading.Thread | None = None
        self._updates: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def is_streaming(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def submit_order(self, order: Order) -> Order:
        """Protocol-compatible order submission."""
        request = _build_alpaca_order_request(order)
        self._client.submit_order(request)
        return order

    def submit_market_order(
        self,
        symbol: str,
        side: Side | int,
        qty: int,
        *,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Any:
        order = Order(
            symbol=str(symbol).upper(),
            side=Side.from_value(side),
            qty=int(qty),
            order_type=OrderType.MARKET,
            time_in_force=time_in_force,
        )
        return self._client.submit_order(_build_alpaca_order_request(order))

    def submit_limit_order(
        self,
        symbol: str,
        side: Side | int,
        qty: int,
        limit_price: float,
        *,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Any:
        order = Order(
            symbol=str(symbol).upper(),
            side=Side.from_value(side),
            qty=int(qty),
            order_type=OrderType.LIMIT,
            time_in_force=time_in_force,
            limit_price=float(limit_price),
        )
        return self._client.submit_order(_build_alpaca_order_request(order))

    def get_positions(self) -> list[Position]:
        """Return current Alpaca positions mapped into project domain types."""
        out: list[Position] = []
        for position in self._client.get_all_positions():
            qty_raw = float(position.qty)
            side = Side.LONG if qty_raw >= 0 else Side.SHORT
            out.append(
                Position(
                    symbol=str(position.symbol).upper(),
                    side=side,
                    qty=abs(int(round(qty_raw))),
                    avg_price=float(position.avg_entry_price),
                )
            )
        return out

    def get_account(self) -> dict[str, Any]:
        """Return a compact account snapshot for notebook display."""
        acct = self._client.get_account()
        if hasattr(acct, "model_dump"):
            data = acct.model_dump()
        elif isinstance(acct, dict):
            data = acct
        else:
            data = acct.__dict__
        keys = ["equity", "cash", "buying_power", "portfolio_value", "status"]
        return {key: data.get(key) for key in keys}

    def cancel_all(self) -> None:
        self._client.cancel_orders()

    def flatten_symbol(self, symbol: str) -> Any | None:
        """Submit the opposing market order to flatten one paper position."""
        symbol = str(symbol).upper()
        positions = {position.symbol: position for position in self.get_positions()}
        if symbol not in positions or positions[symbol].qty == 0:
            return None
        position = positions[symbol]
        exit_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        return self.submit_market_order(symbol, exit_side, position.qty)

    async def _on_trade_update(self, payload: Any) -> None:
        data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
        with self._lock:
            self._updates.append(data)
            self._updates = self._updates[-500:]

    def start_trade_updates(self) -> None:
        """Start the Alpaca trading-stream websocket in a background thread."""
        if self.is_streaming:
            return
        self._stream = TradingStream(
            self.config.alpaca_api_key,
            self.config.alpaca_api_secret,
            paper=self.paper,
            url_override=_optional_stream_url("ALPACA_TRADING_STREAM_URL"),
        )
        self._stream.subscribe_trade_updates(self._on_trade_update)

        def _runner() -> None:
            assert self._stream is not None
            self._stream.run()

        self._thread = threading.Thread(target=_runner, name="alpaca-trade-updates", daemon=True)
        self._thread.start()

    def stop_trade_updates(self, *, timeout: float = 5.0) -> None:
        if self._stream is not None:
            _stop_stream_adapter(self._stream)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._stream = None
        self._thread = None

    def latest_trade_updates(self, *, limit: int = 20) -> pd.DataFrame:
        """Return the most recent streamed trade updates as a dataframe."""
        with self._lock:
            data = list(self._updates[-int(limit):])
        return pd.DataFrame(data)


def notebook_live_monitor(
    market_data: AlpacaLiveMarketData,
    *,
    broker: AlpacaPaperBroker | None = None,
    refresh_seconds: float = 5.0,
    duration_seconds: float | None = 300.0,
    stop_on_exit: bool = False,
) -> None:
    """Live-refresh a notebook chart using Alpaca websocket bars.

    The function blocks until ``duration_seconds`` elapses or the notebook cell
    is interrupted. It is intended for notebook-only monitoring and does not
    affect any backtest state or cached metrics.
    """
    from IPython.display import clear_output, display

    if not market_data.is_running:
        market_data.start()
    deadline = None if duration_seconds is None else time.time() + float(duration_seconds)

    try:
        while deadline is None or time.time() < deadline:
            enriched = market_data.enriched_bars()
            snapshot = compute_live_strategy_snapshot(enriched)
            clear_output(wait=True)
            fig, _ = render_live_strategy_chart(enriched, symbol=market_data.symbol)
            display(fig)
            plt.close(fig)
            display(pd.DataFrame([snapshot]))
            if broker is not None:
                display(pd.DataFrame([broker.get_account()]))
                updates = broker.latest_trade_updates(limit=5)
                if not updates.empty:
                    display(updates.tail(5))
            time.sleep(float(refresh_seconds))
    finally:
        if stop_on_exit:
            market_data.stop()
            if broker is not None:
                broker.stop_trade_updates()
