"""Preprocessing utilities for intraday bar cleanup and day-wise alignment."""

from __future__ import annotations

from datetime import date

import pandas as pd

NY_TZ = "America/New_York"


def _ensure_ny_timestamp(series: pd.Series) -> pd.Series:
    """Ensure timestamps are timezone-aware and converted to America/New_York."""
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize(NY_TZ)
    return ts.dt.tz_convert(NY_TZ)


def preprocess_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and align minute bars to a complete RTH grid per trading day.

    Steps applied:
    - Ensure timestamp is tz-aware in America/New_York
    - Filter regular trading hours 09:30..16:00 (inclusive)
    - Add `date` (YYYY-MM-DD) and `time` (HH:MM)
    - Create complete minute grid per day and left-join bars
    - Forward-fill missing close within each day
    - For missing rows, set open/high/low to close and volume to 0
    - Sort by timestamp

    Args:
        df: Raw minute-bar dataframe with at least timestamp and OHLCV columns.

    Returns:
        pd.DataFrame: Cleaned dataframe with columns timestamp, open, high, low,
            close, volume, date, time.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work["timestamp"] = _ensure_ny_timestamp(work["timestamp"])

    work = work.loc[
        (work["timestamp"].dt.time >= pd.Timestamp("09:30").time())
        & (work["timestamp"].dt.time <= pd.Timestamp("16:00").time())
    ].copy()

    if work.empty:
        empty = work[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        empty["date"] = pd.Series(dtype="string")
        empty["time"] = pd.Series(dtype="string")
        return empty.sort_values("timestamp").reset_index(drop=True)

    work["date"] = work["timestamp"].dt.strftime("%Y-%m-%d")

    grids: list[pd.DataFrame] = []
    for day_str in sorted(work["date"].unique()):
        start_ts = pd.Timestamp(f"{day_str} 09:30", tz=NY_TZ)
        end_ts = pd.Timestamp(f"{day_str} 16:00", tz=NY_TZ)
        grids.append(pd.DataFrame({"timestamp": pd.date_range(start=start_ts, end=end_ts, freq="min")}))

    full_grid = pd.concat(grids, ignore_index=True)
    merged = full_grid.merge(
        work[["timestamp", "open", "high", "low", "close", "volume"]],
        on="timestamp",
        how="left",
    )
    merged["date"] = merged["timestamp"].dt.strftime("%Y-%m-%d")
    merged["time"] = merged["timestamp"].dt.strftime("%H:%M")
    merged["close"] = merged.groupby("date", sort=False)["close"].ffill()
    merged["open"] = merged["open"].fillna(merged["close"])
    merged["high"] = merged["high"].fillna(merged["close"])
    merged["low"] = merged["low"].fillna(merged["close"])
    merged["volume"] = merged["volume"].fillna(0)
    return merged.sort_values("timestamp").reset_index(drop=True)[
        ["timestamp", "open", "high", "low", "close", "volume", "date", "time"]
    ]


def split_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split a bars dataframe into per-day dataframes keyed by `datetime.date`."""
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    work = df.copy()
    work["timestamp"] = _ensure_ny_timestamp(work["timestamp"])
    work = work.sort_values("timestamp")

    out: dict[date, pd.DataFrame] = {}
    for day, day_df in work.groupby(work["timestamp"].dt.date, sort=True):
        out[day] = day_df.reset_index(drop=True)
    return out


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for `preprocess_bars`."""
    return preprocess_bars(df)


def filter_regular_trading_hours(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Backward-compatible helper to filter regular trading hours only."""
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    if work["timestamp"].dt.tz is None:
        work["timestamp"] = work["timestamp"].dt.tz_localize(timezone)
    else:
        work["timestamp"] = work["timestamp"].dt.tz_convert(timezone)

    return work.loc[
        (work["timestamp"].dt.time >= pd.Timestamp("09:30").time())
        & (work["timestamp"].dt.time <= pd.Timestamp("16:00").time())
    ].sort_values("timestamp").reset_index(drop=True)
