"""Robust ML overlay sizing utilities.

These helpers convert raw model output into a more stable sizing signal. The
core idea is to avoid treating absolute probabilities as stationary across time.
Under model drift, calibration can shift even when relative ordering remains
useful. Rank-based sizing preserves the ordering information while reducing
sensitivity to absolute probability scale.

Shrinkage toward one dampens noisy exposure changes. This usually reduces
turnover and drawdown because small probability differences do not translate
into large notional changes.

Overlay gating checks whether the model still has cross-sectional ranking power
in recent history. When the top-decile minus bottom-decile PnL spread turns
non-positive, the overlay is disabled so it cannot keep harming performance.

Risk-aware caps are a final guardrail. When the current intraday regime is more
volatile than normal, the overlay is allowed to modulate exposure only inside a
narrower range so it does not size up aggressively into turbulent conditions.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _percentile_against_prior(value: float, prior: np.ndarray) -> float:
    prior = np.asarray(prior, dtype=float)
    prior = prior[~np.isnan(prior)]
    if prior.size == 0 or np.isnan(value):
        return float('nan')
    less = float(np.sum(prior < value))
    equal = float(np.sum(prior == value))
    return (less + 0.5 * equal) / float(prior.size)


def rolling_rank_percentile(series: pd.Series, window: int) -> pd.Series:
    """Compute a no-lookahead rolling percentile rank.

    For each observation, the percentile rank is computed against the *prior*
    `window` values only. The current value is never included in its own rank,
    which avoids lookahead bias. If fewer than `window` valid prior values are
    available, the result is `NaN`.
    """
    s = pd.Series(series, copy=False, dtype=float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    values = s.to_numpy(dtype=float)
    for i, value in enumerate(values):
        if i < window or np.isnan(value):
            continue
        prior = values[i - window:i]
        prior = prior[~np.isnan(prior)]
        if prior.size < window:
            continue
        out.iloc[i] = _percentile_against_prior(float(value), prior)
    return out


def size_map_with_neutral_zone(
    r: float,
    floor: float,
    cap: float,
    neutral_lo: float,
    neutral_hi: float,
) -> float:
    """Map a percentile rank into a clipped size multiplier.

    The middle of the rank distribution is treated as noise and mapped to
    `1.0`. Lower-percentile signals scale toward `floor`; higher-percentile
    signals scale toward `cap`.
    """
    r = float(np.clip(r, 0.0, 1.0))
    floor = float(floor)
    cap = float(cap)
    neutral_lo = float(neutral_lo)
    neutral_hi = float(neutral_hi)
    if neutral_lo > neutral_hi:
        raise ValueError('neutral_lo must be <= neutral_hi')
    if r >= neutral_lo and r <= neutral_hi:
        return 1.0
    if r < neutral_lo:
        if neutral_lo <= 0:
            return 1.0
        m = floor + (r / neutral_lo) * (1.0 - floor)
        return float(np.clip(m, floor, cap))
    if neutral_hi >= 1:
        return 1.0
    m = 1.0 + ((r - neutral_hi) / (1.0 - neutral_hi)) * (cap - 1.0)
    return float(np.clip(m, floor, cap))


def shrink_toward_one(m_raw: float, lam: float) -> float:
    """Shrink a raw multiplier toward one.

    Shrinkage reduces sensitivity to noisy ranking differences. With `lam=0`,
    the result is exactly `1.0`; with `lam=1`, the raw multiplier is used.
    """
    return float(1.0 + float(lam) * (float(m_raw) - 1.0))


def strategy_vol_managed_multiplier(
    prior_strategy_returns: Iterable[float] | pd.Series,
    lookback_days: int = 20,
    floor: float = 0.5,
    cap: float = 1.5,
) -> tuple[float, float, float]:
    """Compute a no-lookahead strategy-volatility managed exposure multiplier.

    This follows the spirit of Barroso & Santa-Clara: when the strategy's own
    recent realized volatility is high relative to its typical level, reduce
    exposure; when it is low, allow more exposure. Using the strategy's own
    realized volatility is often more effective than using market volatility
    alone because it reacts to the actual instability of the traded signal.

    Only prior returns are used. The returned tuple is:
    `(multiplier, target_vol, current_vol)`.

    The target volatility is the median of prior rolling realized volatility,
    while the current volatility is the latest rolling realized volatility.
    The multiplier is clipped to `[floor, cap]` so the overlay cannot create
    extreme notional swings. If there is insufficient history, the function
    returns a neutral multiplier of `1.0`.
    """
    s = pd.Series(prior_strategy_returns, dtype=float).dropna()
    lookback_days = int(lookback_days)
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")
    if floor <= 0 or cap <= 0 or floor > cap:
        raise ValueError("floor/cap must be positive and satisfy floor <= cap")

    if len(s) < lookback_days + 5:
        return 1.0, float("nan"), float("nan")

    rv = s.rolling(lookback_days).std(ddof=0).dropna()
    if rv.empty:
        return 1.0, float("nan"), float("nan")

    current_vol = float(rv.iloc[-1])
    target_vol = float(rv.median())
    if pd.isna(current_vol) or current_vol <= 0 or pd.isna(target_vol) or target_vol <= 0:
        return 1.0, target_vol, current_vol

    mult = float(np.clip(target_vol / current_vol, float(floor), float(cap)))
    return mult, target_vol, current_vol


def market_vol_managed_multiplier(
    prior_market_returns: Iterable[float] | pd.Series,
    lookback_days: int = 20,
    floor: float = 0.5,
    cap: float = 1.5,
) -> tuple[float, float, float]:
    """Compute a no-lookahead market-volatility managed exposure multiplier.

    This follows the spirit of Moreira & Muir: exposure is reduced when recent
    market volatility is high relative to its typical level and increased when
    recent market volatility is low. The input should contain only prior daily
    market returns, never the current day's return.

    The returned tuple is `(multiplier, target_vol, current_vol)`. The target
    volatility is the median of prior rolling realized market volatility; the
    current volatility is the latest rolling realized market volatility. The
    multiplier is clipped to `[floor, cap]` so exposure cannot jump to extreme
    levels on quiet markets.
    """
    s = pd.Series(prior_market_returns, dtype=float).dropna()
    lookback_days = int(lookback_days)
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")
    if floor <= 0 or cap <= 0 or floor > cap:
        raise ValueError("floor/cap must be positive and satisfy floor <= cap")

    if len(s) < lookback_days + 5:
        return 1.0, float("nan"), float("nan")

    rv = s.rolling(lookback_days).std(ddof=0).dropna()
    if rv.empty:
        return 1.0, float("nan"), float("nan")

    current_vol = float(rv.iloc[-1])
    target_vol = float(rv.median())
    if pd.isna(current_vol) or current_vol <= 0 or pd.isna(target_vol) or target_vol <= 0:
        return 1.0, target_vol, current_vol

    mult = float(np.clip(target_vol / current_vol, float(floor), float(cap)))
    return mult, target_vol, current_vol


def panic_derisk_multiplier(
    prior_closes: Iterable[float] | pd.Series,
    prior_market_returns: Iterable[float] | pd.Series,
    return_lookback_days: int = 20,
    vol_lookback_days: int = 20,
    vol_quantile: float = 0.8,
    panic_exposure: float = 0.5,
) -> tuple[float, float, float, float]:
    """Compute a no-lookahead panic-state de-risk multiplier.

    This follows the Daniel & Moskowitz intuition: momentum-like strategies are
    most vulnerable after recent drawdowns when volatility is already elevated.
    If the prior `return_lookback_days` return is negative *and* current market
    volatility is above a high historical threshold, the function returns a
    reduced exposure (`panic_exposure`); otherwise it returns `1.0`.
    """
    closes = pd.Series(prior_closes, dtype=float).dropna()
    rets = pd.Series(prior_market_returns, dtype=float).dropna()
    if not (0.0 < float(vol_quantile) < 1.0):
        raise ValueError("vol_quantile must be in (0, 1)")
    if panic_exposure <= 0:
        raise ValueError("panic_exposure must be positive")
    if len(closes) < int(return_lookback_days) + 1 or len(rets) < int(vol_lookback_days) + 5:
        return 1.0, float("nan"), float("nan"), float("nan")

    current_ret = float(closes.iloc[-1] / closes.iloc[-1 - int(return_lookback_days)] - 1.0)
    rv = rets.rolling(int(vol_lookback_days)).std(ddof=0).dropna()
    if rv.empty:
        return 1.0, current_ret, float("nan"), float("nan")

    current_vol = float(rv.iloc[-1])
    vol_threshold = float(rv.quantile(float(vol_quantile)))
    if pd.isna(current_vol) or pd.isna(vol_threshold):
        return 1.0, current_ret, current_vol, vol_threshold

    mult = float(panic_exposure) if (current_ret < 0.0 and current_vol >= vol_threshold) else 1.0
    return mult, current_ret, current_vol, vol_threshold


def trend_state_multiplier(
    prior_closes: Iterable[float] | pd.Series,
    lookback_days: int = 60,
    low_exposure: float = 0.85,
    high_exposure: float = 1.15,
) -> tuple[float, float, float]:
    """Compute a no-lookahead trend-state multiplier.

    This follows the time-series momentum intuition from Moskowitz, Ooi, and
    Pedersen: stronger medium-horizon trend states deserve somewhat higher
    exposure than weak trend states. The current absolute lookback return is
    compared with the median absolute lookback return observed in prior history.
    """
    closes = pd.Series(prior_closes, dtype=float).dropna()
    lookback_days = int(lookback_days)
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")
    if low_exposure <= 0 or high_exposure <= 0:
        raise ValueError("trend-state exposures must be positive")
    if len(closes) < (2 * lookback_days + 1):
        return 1.0, float("nan"), float("nan")

    current_ret = float(closes.iloc[-1] / closes.iloc[-1 - lookback_days] - 1.0)
    hist_abs = (closes / closes.shift(lookback_days) - 1.0).abs().dropna()
    if hist_abs.empty:
        return 1.0, current_ret, float("nan")

    threshold = float(hist_abs.median())
    mult = float(high_exposure) if abs(current_ret) >= threshold else float(low_exposure)
    return mult, current_ret, threshold


def fast_alpha_tactical_multiplier(
    side: int,
    recent_return: float,
    favorable_mult: float = 1.15,
    unfavorable_mult: float = 0.85,
) -> tuple[float, bool]:
    """Map short-horizon alpha direction into a tactical entry multiplier.

    This is a minimal source-level implementation of the "Fast Alphas" idea:
    short-term moves against the intended trend direction are treated as
    favorable tactical entry states because they provide a local pullback into
    the larger breakout signal. Long entries therefore prefer recent negative
    returns; short entries prefer recent positive returns.
    """
    if side not in (-1, 1) or pd.isna(recent_return):
        return 1.0, False
    if favorable_mult <= 0 or unfavorable_mult <= 0:
        raise ValueError("favorable_mult/unfavorable_mult must be positive")
    favorable = (side > 0 and float(recent_return) < 0.0) or (side < 0 and float(recent_return) > 0.0)
    return float(favorable_mult if favorable else unfavorable_mult), bool(favorable)


def execution_aware_entry_multiplier(
    side: int,
    signal_price: float,
    execution_price: float,
    *,
    max_adverse_bps: float = 8.0,
    adverse_mult: float = 0.5,
) -> tuple[float, float]:
    """Downsize entries when the executable price drifts too far adversely.

    This is a conservative proxy for a marketable-limit participation rule.
    The function uses the signal-bar close as the reference price and the
    actual executable price (typically the next-minute open) as the realized
    fill level. If the fill drifts too far against the intended trade
    direction, exposure is reduced.

    Returns `(multiplier, adverse_bps)`.
    """
    if int(side) not in (-1, 1):
        return 1.0, 0.0
    signal_price = float(signal_price)
    execution_price = float(execution_price)
    if (
        signal_price <= 0.0
        or execution_price <= 0.0
        or float(max_adverse_bps) <= 0.0
        or float(adverse_mult) <= 0.0
    ):
        return 1.0, 0.0
    if int(side) > 0:
        adverse_bps = max((execution_price / signal_price - 1.0) * 10000.0, 0.0)
    else:
        adverse_bps = max((signal_price / execution_price - 1.0) * 10000.0, 0.0)
    if adverse_bps >= float(max_adverse_bps):
        return float(adverse_mult), float(adverse_bps)
    return 1.0, float(adverse_bps)


def execution_aware_relative_entry_multiplier(
    side: int,
    signal_price: float,
    execution_price: float,
    *,
    band_width_pct: float,
    realized_vol_30m: float,
    max_adverse_band_frac: float = 0.15,
    max_adverse_vol_mult: float = 0.75,
    adverse_mult: float = 0.5,
) -> tuple[float, float, float]:
    """Downsize entries when adverse drift is large relative to local risk.

    Raw bps thresholds are often too blunt for intraday breakout entries.
    This version scales the allowable adverse drift by the current band width
    and recent intraday volatility, then downsizes only when the realized
    drift exceeds that context-aware threshold.

    Returns `(multiplier, adverse_return, threshold_return)`.
    """
    if int(side) not in (-1, 1):
        return 1.0, 0.0, 0.0
    signal_price = float(signal_price)
    execution_price = float(execution_price)
    if signal_price <= 0.0 or execution_price <= 0.0 or float(adverse_mult) <= 0.0:
        return 1.0, 0.0, 0.0
    if int(side) > 0:
        adverse_return = max(execution_price / signal_price - 1.0, 0.0)
    else:
        adverse_return = max(signal_price / execution_price - 1.0, 0.0)
    band_width_pct = max(float(band_width_pct), 0.0)
    realized_vol_30m = max(float(realized_vol_30m), 0.0)
    threshold_return = max(
        float(max_adverse_band_frac) * band_width_pct,
        float(max_adverse_vol_mult) * realized_vol_30m,
    )
    if threshold_return > 0.0 and adverse_return >= threshold_return:
        return float(adverse_mult), float(adverse_return), float(threshold_return)
    return 1.0, float(adverse_return), float(threshold_return)


def intraday_risk_size_multiplier(
    current_value: float,
    prior_values: Iterable[float] | pd.Series,
    *,
    risk_quantile: float = 0.8,
    high_risk_mult: float = 0.75,
    min_history: int = 20,
) -> tuple[float, float]:
    """Downsize exposure when current intraday risk is historically elevated.

    The threshold is computed from prior observations only. This is designed
    for realistic execution settings where high intraday volatility tends to
    worsen fills and increase stop churn.

    Returns `(multiplier, threshold)`.
    """
    if not (0.0 < float(risk_quantile) < 1.0):
        raise ValueError("risk_quantile must be in (0, 1)")
    if float(high_risk_mult) <= 0.0:
        raise ValueError("high_risk_mult must be positive")
    series = pd.Series(prior_values, dtype=float).dropna()
    if len(series) < int(min_history) or pd.isna(current_value):
        return 1.0, float("nan")
    threshold = float(series.quantile(float(risk_quantile)))
    if pd.isna(threshold):
        return 1.0, float("nan")
    mult = float(high_risk_mult) if float(current_value) >= threshold else 1.0
    return mult, threshold




def convex_rank_bucket_map(
    r: float,
    low_cut: float = 0.40,
    high_cut: float = 0.80,
    low_mult: float = 0.70,
    mid_mult: float = 1.00,
    high_mult: float = 2.00,
    convex_cap_mult: float = 2.50,
) -> tuple[float, str]:
    """Map a rank percentile into a convex three-bucket size multiplier.

    Rank-based sizing is more stable than absolute probability cutoffs because
    score scales can drift even when rank ordering remains informative. This
    helper intentionally concentrates risk into the highest-ranked setups while
    down-weighting the weakest ones.
    """
    if pd.isna(r):
        return 1.0, "mid"
    r = float(np.clip(r, 0.0, 1.0))
    if r < float(low_cut):
        mult, bucket = float(low_mult), "low"
    elif r >= float(high_cut):
        mult, bucket = float(high_mult), "high"
    else:
        mult, bucket = float(mid_mult), "mid"
    return float(np.clip(mult, 0.0, float(convex_cap_mult))), bucket


def risk_state_multiplier(
    risk_value: float,
    risk_threshold: float,
    floor: float,
    cap: float,
    high_risk_floor: float,
    high_risk_cap: float,
) -> tuple[float, float]:
    """Return effective multiplier bounds under the current risk state."""
    if pd.isna(risk_value) or pd.isna(risk_threshold):
        return (float(floor), float(cap))
    if float(risk_value) >= float(risk_threshold):
        return (float(high_risk_floor), float(high_risk_cap))
    return (float(floor), float(cap))


def compute_overlay_enabled_flag(
    df_recent: pd.DataFrame,
    score_col: str,
    pnl_col: str,
    lookback_days: int,
) -> bool:
    """Return whether the overlay should stay enabled.

    Recent observations are ranked by `score_col`, split into deciles, and the
    top-decile minus bottom-decile mean PnL spread is computed. A positive
    spread indicates that the model still has useful ranking power; otherwise
    the overlay is disabled.

    If the sample is too small or too degenerate to form meaningful deciles,
    the function returns `True` and logs a warning rather than disabling the
    overlay on fragile evidence.
    """
    if df_recent is None or df_recent.empty:
        logger.warning('Overlay gating: no recent samples, defaulting overlay to enabled.')
        return True

    d = df_recent.copy()
    ts_col = None
    for cand in ('exit_timestamp', 'timestamp'):
        if cand in d.columns:
            ts_col = cand
            break
    if ts_col is not None:
        ts = pd.to_datetime(d[ts_col])
        days = pd.Series(ts.dt.strftime('%Y-%m-%d'))
    elif 'date' in d.columns:
        days = pd.Series(pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d'))
    else:
        logger.warning('Overlay gating: no timestamp/date column, defaulting overlay to enabled.')
        return True

    d = d.assign(__day=days.values)
    unique_days = d['__day'].dropna().drop_duplicates().sort_values().tolist()
    if len(unique_days) > int(lookback_days):
        keep = set(unique_days[-int(lookback_days):])
        d = d.loc[d['__day'].isin(keep)].copy()

    d = d.loc[d[score_col].notna() & d[pnl_col].notna()].copy()
    if len(d) < 20:
        logger.warning('Overlay gating: only %d recent samples, defaulting overlay to enabled.', len(d))
        return True

    try:
        d['decile'] = pd.qcut(d[score_col], 10, labels=False, duplicates='drop')
    except Exception:
        logger.warning('Overlay gating: decile assignment failed, defaulting overlay to enabled.')
        return True

    if d['decile'].nunique() < 2:
        logger.warning('Overlay gating: insufficient score dispersion, defaulting overlay to enabled.')
        return True

    top = d.loc[d['decile'] == d['decile'].max(), pnl_col].mean()
    bot = d.loc[d['decile'] == d['decile'].min(), pnl_col].mean()
    spread = float(top - bot)
    return bool(spread > 0)
