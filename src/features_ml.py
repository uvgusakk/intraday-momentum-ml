"""Feature and target construction for ML classification from baseline candidates."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_strategy import compute_break_strength, run_baseline_backtest
from .config import load_config

logger = logging.getLogger(__name__)
NY_TZ = "America/New_York"
BASE_FEATURE_COLS = [
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


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume", "UB", "LB", "VWAP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ)
    else:
        ts = ts.dt.tz_convert(NY_TZ)

    out["timestamp"] = ts
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["time"] = out["timestamp"].dt.strftime("%H:%M")
    return out


def _compute_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    f = bars.copy()
    f["open_0930"] = f.groupby("date", sort=False)["open"].transform("first")
    f["midband"] = (f["UB"] + f["LB"]) / 2.0

    f["intraday_return"] = f["close"] / f["open_0930"] - 1.0
    f["close_30m_ago"] = f.groupby("date", sort=False)["close"].shift(30)
    f["ret_30m"] = f["close"] / f["close_30m_ago"] - 1.0

    f["ret_1m"] = f.groupby("date", sort=False)["close"].pct_change()
    f["realized_vol_30m"] = (
        f.groupby("date", sort=False)["ret_1m"]
        .transform(lambda s: s.rolling(window=30, min_periods=30).std(ddof=0))
        .astype(float)
    )

    signed_mid = np.sign(f["close"] - f["midband"]).replace(0, np.nan)
    signed_mid = signed_mid.groupby(f["date"], sort=False).ffill().fillna(0)
    flips = ((signed_mid * signed_mid.groupby(f["date"], sort=False).shift(1)) < 0).astype(int)
    f["whipsaw_60m"] = flips.groupby(f["date"], sort=False).transform(
        lambda s: s.rolling(window=60, min_periods=1).sum()
    )

    minutes = (
        f["timestamp"].dt.hour * 60
        + f["timestamp"].dt.minute
        - (9 * 60 + 30)
    ).astype(int)
    f["time_of_day_minutes"] = minutes

    angle = 2.0 * np.pi * (f["time_of_day_minutes"] / 390.0)
    f["tod_sin"] = np.sin(angle)
    f["tod_cos"] = np.cos(angle)

    f["band_width"] = (f["UB"] - f["LB"]) / f["open_0930"]
    f["vwap_diff"] = (f["close"] - f["VWAP"]) / f["VWAP"]
    f["break_strength"] = f.apply(compute_break_strength, axis=1)

    return f


def _extract_candidates_with_labels(
    bars: pd.DataFrame,
    backtest_kwargs: dict | None = None,
) -> pd.DataFrame:
    kwargs = backtest_kwargs or {}
    bt = run_baseline_backtest(bars, **kwargs)
    trades = bt["trades"].copy()

    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "side", "y", "pnl", "trade_return", "costs"])

    decision_col = "decision_timestamp" if "decision_timestamp" in trades.columns else "entry_timestamp"
    trades["timestamp"] = pd.to_datetime(trades[decision_col])
    if trades["timestamp"].dt.tz is None:
        trades["timestamp"] = trades["timestamp"].dt.tz_localize(NY_TZ)
    else:
        trades["timestamp"] = trades["timestamp"].dt.tz_convert(NY_TZ)
    trades["side"] = trades["side"].map({"long": 1, "short": -1})
    trades["y"] = (trades["pnl"] > 0).astype(int)
    gross_notional = (trades["entry_price"].abs() * trades["shares"]).replace(0, np.nan)
    trades["trade_return"] = (trades["pnl"] / gross_notional).fillna(0.0)

    cols = ["timestamp", "side", "y", "pnl", "trade_return", "costs"]
    return trades[cols].drop_duplicates().reset_index(drop=True)


def _extract_candidates_fixed_horizon(
    bars: pd.DataFrame,
    backtest_kwargs: dict | None = None,
    horizon_mins: int = 30,
) -> pd.DataFrame:
    """Create candidate entries with fixed-horizon net-return labels."""
    if horizon_mins < 1:
        raise ValueError("horizon_mins must be >= 1")

    kwargs = backtest_kwargs or {}
    bt = run_baseline_backtest(bars, **kwargs)
    trades = bt["trades"].copy()
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "side", "y", "pnl", "trade_return", "costs"])

    c = trades.copy()
    decision_col = "decision_timestamp" if "decision_timestamp" in c.columns else "entry_timestamp"
    c["timestamp"] = pd.to_datetime(c[decision_col])
    if c["timestamp"].dt.tz is None:
        c["timestamp"] = c["timestamp"].dt.tz_localize(NY_TZ)
    else:
        c["timestamp"] = c["timestamp"].dt.tz_convert(NY_TZ)
    c["side"] = c["side"].map({"long": 1, "short": -1})
    px = bars[["timestamp", "date", "close"]].copy().sort_values("timestamp")
    px["close_fwd"] = px.groupby("date", sort=False)["close"].shift(-horizon_mins)

    labeled = c[["timestamp", "side"]].merge(
        px[["timestamp", "close", "close_fwd"]],
        on=["timestamp"],
        how="left",
        validate="many_to_one",
    )
    labeled = labeled.dropna(subset=["close", "close_fwd"]).copy()

    commission = float(kwargs.get("commission_per_share", 0.0035))
    slippage = float(kwargs.get("slippage_per_share", 0.001))
    spread_bps = float(kwargs.get("spread_bps", 0.0))
    roundtrip_cost_ret = (
        2.0 * (commission + slippage)
        + (spread_bps / 10000.0) * labeled["close"]
    ) / labeled["close"].replace(0, np.nan)

    gross_ret = labeled["side"] * (labeled["close_fwd"] / labeled["close"] - 1.0)
    net_ret = (gross_ret - roundtrip_cost_ret).fillna(0.0)

    labeled["trade_return"] = net_ret
    labeled["pnl"] = net_ret
    labeled["costs"] = roundtrip_cost_ret.fillna(0.0)
    labeled["y"] = (labeled["trade_return"] > 0).astype(int)

    cols = ["timestamp", "side", "y", "pnl", "trade_return", "costs"]
    return labeled[cols].drop_duplicates().reset_index(drop=True)


def build_ml_dataset(
    df: pd.DataFrame,
    backtest_kwargs: dict | None = None,
    label_mode: str = "fixed_horizon",
    horizon_mins: int = 30,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build ML dataset from baseline candidate entries.

    Args:
        df: Long intraday bars dataframe with UB/LB/VWAP already computed.
        backtest_kwargs: Optional kwargs forwarded to run_baseline_backtest.
        label_mode: ``baseline_trade`` or ``fixed_horizon``.
        horizon_mins: Forward horizon for ``fixed_horizon`` labels.
        persist: When True, save the dataset to ``DATA_DIR/ml_dataset.parquet``.

    Returns:
        X feature matrix, y labels, meta columns (date, timestamp, side, symbol, pnl...).
    """
    if label_mode not in {"baseline_trade", "fixed_horizon"}:
        raise ValueError("label_mode must be one of: 'baseline_trade', 'fixed_horizon'")

    bars = _normalize_bars(df)
    feat = _compute_feature_frame(bars)
    if label_mode == "fixed_horizon":
        candidates = _extract_candidates_fixed_horizon(
            bars, backtest_kwargs=backtest_kwargs, horizon_mins=horizon_mins
        )
    else:
        candidates = _extract_candidates_with_labels(bars, backtest_kwargs=backtest_kwargs)

    meta_cols = ["date", "timestamp", "side", "pnl", "trade_return", "costs"]

    if candidates.empty:
        logger.warning("No baseline candidate entries found; returning empty dataset.")
        X_empty = pd.DataFrame(columns=list(BASE_FEATURE_COLS))
        y_empty = pd.Series(dtype=int, name="y")
        meta_empty = pd.DataFrame(columns=meta_cols)
        if persist:
            _save_ml_dataset(X_empty, y_empty, meta_empty)
        return X_empty, y_empty, meta_empty

    merged = candidates.merge(
        feat,
        on=["timestamp"],
        how="left",
        validate="one_to_one",
    )

    merged["signed_break_distance"] = np.where(
        merged["side"] > 0,
        (merged["close"] - merged["UB"]) / merged["UB"],
        (merged["LB"] - merged["close"]) / merged["LB"],
    )

    merged = merged.dropna(subset=BASE_FEATURE_COLS + ["y"]).reset_index(drop=True)

    X = merged[BASE_FEATURE_COLS].copy()
    y = merged["y"].astype(int).rename("y")
    meta = merged[meta_cols].copy()

    if persist:
        _save_ml_dataset(X, y, meta)
    return X, y, meta


def _save_ml_dataset(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> Path:
    """Persist the ML dataset to DATA_DIR/ml_dataset.parquet."""
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.data_dir / "ml_dataset.parquet"

    save_df = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    save_df.to_parquet(out_path, index=False)
    logger.info("Saved ML dataset to %s (%d rows)", out_path, len(save_df))
    return out_path


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible helper returning only engineered features."""
    X, _, _ = build_ml_dataset(df)
    return X


def build_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Backward-compatible helper returning target labels from ML dataset.

    `horizon` is unused here and kept for signature compatibility.
    """
    _ = horizon
    _, y, _ = build_ml_dataset(df)
    return y
