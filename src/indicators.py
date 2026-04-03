"""Technical indicator calculations for intraday momentum workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"


def _normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp/date/time helper columns exist and are normalized."""
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ)
    else:
        ts = ts.dt.tz_convert(NY_TZ)

    out["timestamp"] = ts
    if "date" not in out.columns:
        out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    if "time" not in out.columns:
        out["time"] = out["timestamp"].dt.strftime("%H:%M")

    out = out.sort_values(["date", "timestamp"]).reset_index(drop=True)
    return out


def compute_intraday_move_from_open(df: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute move from each day's 09:30 open for every minute.

    Adds:
        move_abs = abs(close / open_0930 - 1)
    """
    required = {"open", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = _normalize_time_columns(df)

    open_0930_by_day = out.loc[out["time"] == "09:30", ["date", "open"]].rename(columns={"open": "open_0930"})

    if open_0930_by_day.empty:
        raise ValueError("No 09:30 bars found; cannot compute open_0930 anchor.")

    out = out.merge(open_0930_by_day, on=["date"], how="left")

    # Fallback: if a day has no explicit 09:30 bar, use that day's first open.
    out["open_0930"] = out["open_0930"].fillna(
        out.groupby("date", sort=False)["open"].transform("first")
    )

    out["move_abs"] = (out["close"] / out["open_0930"] - 1.0).abs()
    return out


def compute_sigma_profile(df: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
    """Compute minute-of-day sigma as trailing mean of prior-day `move_abs`.

    For each minute-of-day, day d uses only d-1..d-lookback_days.
    """
    if "move_abs" not in df.columns:
        raise ValueError("Missing required column: move_abs. Run compute_intraday_move_from_open first.")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    out = _normalize_time_columns(df)
    out["trade_date"] = pd.to_datetime(out["date"]).dt.date

    # One row per (day, minute) for stable rolling alignment.
    minute_day = (
        out[["trade_date", "time", "move_abs"]]
        .groupby(["trade_date", "time"], as_index=False, sort=True)["move_abs"]
        .last()
    )

    def _roll_prior(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=lookback_days, min_periods=1).mean()

    minute_day["sigma"] = minute_day.groupby("time", sort=False)["move_abs"].transform(_roll_prior)
    out = out.merge(minute_day[["trade_date", "time", "sigma"]], on=["trade_date", "time"], how="left")
    out = out.drop(columns=["trade_date"])
    return out


def compute_gap_adjusted_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap-adjusted upper/lower bands using sigma profile.

    Adds columns:
        UB, LB
    """
    required = {"open", "close", "sigma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = _normalize_time_columns(df)
    out["trade_date"] = pd.to_datetime(out["date"]).dt.date

    day_table = out.groupby(["trade_date"], as_index=False, sort=True).agg(
        open_0930=("open", "first"),
        close_1600=("close", "last"),
    )
    day_table["prevClose"] = day_table["close_1600"].shift(1)
    day_table["anchorUpper"] = np.maximum(day_table["open_0930"], day_table["prevClose"])
    day_table["anchorLower"] = np.minimum(day_table["open_0930"], day_table["prevClose"])

    out = out.merge(
        day_table[["trade_date", "anchorUpper", "anchorLower"]],
        on=["trade_date"],
        how="left",
    )

    out["UB"] = out["anchorUpper"] * (1.0 + out["sigma"])
    out["LB"] = out["anchorLower"] * (1.0 - out["sigma"])

    out = out.drop(columns=["trade_date", "anchorUpper", "anchorLower"])
    return out


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday VWAP per day with division-by-zero guard."""
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = _normalize_time_columns(df)
    out["tp"] = (out["high"] + out["low"] + out["close"]) / 3.0

    pv = out["tp"] * out["volume"]
    cum_pv = pv.groupby(out["date"], sort=False).cumsum()
    cum_vol = out["volume"].groupby(out["date"], sort=False).cumsum()

    out["VWAP"] = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)
    out = out.drop(columns=["tp"])
    return out


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for VWAP calculation."""
    return compute_vwap(df)


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add basic rolling mean/std features for close and volume.

    This helper is kept lightweight for compatibility with earlier scaffold code.
    """
    out = _normalize_time_columns(df)
    for w in windows:
        if w < 1:
            raise ValueError("All windows must be >= 1")
        out[f"close_mean_{w}"] = out.groupby("date", sort=False)["close"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        out[f"close_std_{w}"] = out.groupby("date", sort=False)["close"].transform(
            lambda s: s.rolling(w, min_periods=1).std(ddof=0)
        )
    return out
