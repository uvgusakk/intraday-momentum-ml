"""Data access helpers for fetching Alpaca intraday bar data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .config import load_config

logger = logging.getLogger(__name__)


def _sanitize_for_filename(value: str) -> str:
    """Sanitize string values for filesystem-safe filenames."""
    return (
        value.strip()
        .replace(" ", "_")
        .replace(":", "-")
        .replace("/", "-")
        .replace("+", "plus")
    )


def _parse_to_utc(ts_value: str, tz_name: str) -> pd.Timestamp:
    """Parse timestamp input and convert it to UTC."""
    ts = pd.Timestamp(ts_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_name)
    return ts.tz_convert("UTC")


def _build_cache_path(symbol: str, start: str, end: str, data_dir: Path) -> Path:
    """Build parquet cache path for a symbol and date interval."""
    start_key = _sanitize_for_filename(start)
    end_key = _sanitize_for_filename(end)
    filename = f"{symbol.upper()}_1min_{start_key}_{end_key}.parquet"
    return data_dir / filename


def _iter_chunks(start_utc: pd.Timestamp, end_utc: pd.Timestamp, chunk_days: int = 5):
    """Yield UTC time chunks to avoid provider-side per-request row caps."""
    cursor = start_utc
    delta = pd.Timedelta(days=chunk_days)
    while cursor < end_utc:
        nxt = min(cursor + delta, end_utc)
        yield cursor, nxt
        cursor = nxt


def _cache_covers_interval(
    bars: pd.DataFrame,
    start_local: pd.Timestamp,
    end_local: pd.Timestamp,
) -> bool:
    """Return True if cached bars likely cover requested [start, end) interval."""
    if bars.empty:
        return False
    min_ts = pd.to_datetime(bars["timestamp"]).min()
    max_ts = pd.to_datetime(bars["timestamp"]).max()

    # Allow small tolerance because ranges can start/end outside market hours.
    lower_ok = min_ts <= start_local + pd.Timedelta(days=2)
    upper_ok = max_ts >= end_local - pd.Timedelta(days=2)
    return bool(lower_ok and upper_ok)


def fetch_minute_bars(
    symbol: str,
    start: str,
    end: str,
    adjustment: str = "raw",
    force: bool = False,
) -> pd.DataFrame:
    """Fetch 1-minute bars within [start, end), with parquet caching.

    Args:
        symbol: Ticker symbol (e.g., "SPY").
        start: Interval start timestamp string.
        end: Interval end timestamp string (exclusive in returned dataframe).
        adjustment: Alpaca adjustment mode (e.g., "raw", "split", "dividend", "all").
        force: If True, ignore cached parquet and refetch from Alpaca.

    Returns:
        pd.DataFrame: Columns are timestamp, open, high, low, close, volume.
    """
    config = load_config()
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    start_utc = _parse_to_utc(start, config.tz)
    end_utc = _parse_to_utc(end, config.tz)

    if end_utc <= start_utc:
        raise ValueError("`end` must be after `start`.")

    start_local = start_utc.tz_convert(config.tz)
    end_local = end_utc.tz_convert(config.tz)

    cache_path = _build_cache_path(symbol=symbol, start=start, end=end, data_dir=data_dir)

    if cache_path.exists() and not force:
        logger.info("Loading cached bars from %s", cache_path)
        cached = load_bars_parquet(str(cache_path), tz_name=config.tz)
        if _cache_covers_interval(cached, start_local=start_local, end_local=end_local):
            return cached
        logger.warning(
            "Cached bars appear incomplete for [%s, %s). Refetching with force.",
            start,
            end,
        )

    logger.info(
        "Fetching 1-minute bars from Alpaca symbol=%s start=%s end=%s adjustment=%s",
        symbol,
        start_utc.isoformat(),
        end_utc.isoformat(),
        adjustment,
    )

    try:
        client = StockHistoricalDataClient(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_api_secret,
            url_override=config.alpaca_data_url,
        )

        pieces: list[pd.DataFrame] = []
        for chunk_start, chunk_end in _iter_chunks(start_utc, end_utc, chunk_days=5):
            request = StockBarsRequest(
                symbol_or_symbols=[symbol.upper()],
                timeframe=TimeFrame.Minute,
                start=chunk_start.to_pydatetime(),
                end=chunk_end.to_pydatetime(),
                adjustment=adjustment,
                limit=10_000,
            )
            response = client.get_stock_bars(request)
            raw = response.df.reset_index()

            if raw.empty:
                logger.info(
                    "Chunk %s -> %s returned 0 bars.",
                    chunk_start.isoformat(),
                    chunk_end.isoformat(),
                )
                continue

            raw = raw[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            pieces.append(raw)
            logger.info(
                "Chunk %s -> %s returned %d bars.",
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                len(raw),
            )

        if not pieces:
            logger.warning("No bars returned for symbol=%s in requested range.", symbol)
            empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            save_bars_parquet(empty, str(cache_path))
            return empty

        bars = pd.concat(pieces, ignore_index=True)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(config.tz)

        # Enforce [start, end) contract and deduplicate chunk boundaries.
        bars = bars.loc[(bars["timestamp"] >= start_local) & (bars["timestamp"] < end_local)]
        bars = bars.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        save_bars_parquet(bars, str(cache_path))
        logger.info("Saved %s bars to %s", len(bars), cache_path)
        return bars

    except Exception as exc:
        logger.exception("Failed to fetch bars from Alpaca for %s: %s", symbol, exc)
        raise RuntimeError(f"Failed to fetch minute bars for {symbol}") from exc


def save_bars_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Save bar data to a Parquet file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_bars_parquet(path: str, tz_name: str = "America/New_York") -> pd.DataFrame:
    """Load bar data from a Parquet file and normalize schema/timezone."""
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("Parquet file missing required 'timestamp' column.")

    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df["timestamp"] = ts.dt.tz_convert(tz_name)

    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Parquet file missing required columns: {missing}")

    return df[expected].sort_values("timestamp").reset_index(drop=True)
