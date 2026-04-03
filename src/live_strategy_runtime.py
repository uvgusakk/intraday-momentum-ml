"""Separate live/websocket runtime for paper monitoring and routing.

This module is intentionally additive:

- it does not modify the score-forward or notebook research path;
- it reuses the existing baseline signal, ML scoring, and hybrid-stop helpers;
- it can monitor all live variants at once, while paper routing is done for one
  selected variant at a time.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .backtest_ml_filter import (
    _align_model_features,
    _build_feature_row,
    _hybrid_stop_trigger_details,
    _load_artifact_bundle,
    _raw_scores,
)
from .baseline_strategy import (
    OpenTrade,
    _compute_daily_sizing_table,
    _stop_triggered,
    stop_trigger_details,
)
from .config import (
    DEFAULT_DECISION_FREQ_MINS,
    DEFAULT_EXECUTION_SPREAD_BPS,
    DEFAULT_MINUTE_STOP_MONITORING,
    DEFAULT_SOFT_SIZE_CAP,
    DEFAULT_SOFT_SIZE_FLOOR,
    DEFAULT_USE_NEXT_BAR_OPEN,
    load_config,
)
from .core.types import Position, Side, Signal
from .live_alpaca import AlpacaLiveMarketData, AlpacaPaperBroker
from .strategies import BaselineNoiseAreaStrategy, MLOutputSizerRiskManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveStrategyVariant:
    """Runtime configuration for one live strategy view or paper trader."""

    name: str
    description: str
    use_ml: bool
    decision_freq_mins: int = DEFAULT_DECISION_FREQ_MINS
    first_trade_time: str = "10:00"
    use_next_bar_open: bool = DEFAULT_USE_NEXT_BAR_OPEN
    minute_stop_monitoring: bool = DEFAULT_MINUTE_STOP_MONITORING
    spread_bps: float = DEFAULT_EXECUTION_SPREAD_BPS
    size_floor: float = DEFAULT_SOFT_SIZE_FLOOR
    size_cap: float = DEFAULT_SOFT_SIZE_CAP
    hybrid_stop_mode: bool = False
    catastrophic_stop_bps: float = 0.0
    regime_overlay: bool = False


LIVE_STRATEGY_VARIANTS: dict[str, LiveStrategyVariant] = {
    "baseline": LiveStrategyVariant(
        name="baseline",
        description="Paper baseline breakout logic with realistic execution settings and no ML sizing.",
        use_ml=False,
    ),
    "soft_hybrid_7_5": LiveStrategyVariant(
        name="soft_hybrid_7_5",
        description="Realistic soft-sized overlay with hybrid catastrophic minute stops at 7.5 bps.",
        use_ml=True,
        hybrid_stop_mode=True,
        catastrophic_stop_bps=7.5,
    ),
    "soft_hybrid_10": LiveStrategyVariant(
        name="soft_hybrid_10",
        description="Realistic soft-sized overlay with hybrid catastrophic minute stops at 10 bps.",
        use_ml=True,
        hybrid_stop_mode=True,
        catastrophic_stop_bps=10.0,
    ),
    "soft_hybrid_5": LiveStrategyVariant(
        name="soft_hybrid_5",
        description="Realistic soft-sized overlay with hybrid catastrophic minute stops at 5 bps.",
        use_ml=True,
        hybrid_stop_mode=True,
        catastrophic_stop_bps=5.0,
    ),
}

DEFAULT_LIVE_VARIANTS = ["baseline", "soft_hybrid_7_5", "soft_hybrid_10", "soft_hybrid_5"]


def get_live_variant(name: str) -> LiveStrategyVariant:
    """Fetch one supported live variant by name."""
    key = str(name).strip()
    if key not in LIVE_STRATEGY_VARIANTS:
        raise ValueError(f"Unsupported live variant: {key}. Allowed: {sorted(LIVE_STRATEGY_VARIANTS)}")
    return LIVE_STRATEGY_VARIANTS[key]


def list_live_variants() -> list[str]:
    """Return the canonical live variant ordering."""
    return list(DEFAULT_LIVE_VARIANTS)


def _safe_calibrated_probability(bundle: Mapping[str, Any], row: pd.Series, desired_side: int) -> float:
    """Score the current candidate row with the trained model bundle."""
    if int(desired_side) == 0:
        return float("nan")
    model = bundle["model"]
    calibrator = bundle["calibrator"]
    x_row = _build_feature_row(row, side=int(desired_side), model=model)
    if x_row.isna().any(axis=None):
        return float("nan")
    x_row = _align_model_features(model, x_row)
    scores = _raw_scores(model, x_row)
    probs = calibrator.predict_proba(scores)
    arr = np.asarray(probs)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return float(arr[0, 0])
        return float(arr[0, 1])
    flat = arr.reshape(-1)
    return float(flat[0]) if flat.size else float("nan")


def _latest_buffer_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    """Return the latest timestamp present in a dataframe buffer."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    return pd.Timestamp(pd.to_datetime(df["timestamp"]).max())


def _latest_account_equity(account_snapshot: Mapping[str, Any] | None, fallback: float) -> float:
    """Extract a usable account equity number from the broker snapshot."""
    if not account_snapshot:
        return float(fallback)
    for key in ("equity", "portfolio_value", "buying_power", "cash"):
        raw = account_snapshot.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return float(fallback)


def _latest_daily_sizing(
    enriched_bars: pd.DataFrame,
    *,
    account_equity: float,
    sigma_target: float,
    lev_cap: float,
) -> tuple[float, float, int]:
    """Compute the latest leverage and baseline share quantity."""
    daily = _compute_daily_sizing_table(enriched_bars, account_equity, sigma_target, lev_cap)
    if daily.empty:
        return 1.0, float("nan"), 0
    latest = daily.iloc[-1]
    leverage = float(latest["leverage"]) if not pd.isna(latest["leverage"]) else min(1.0, lev_cap)
    day_open = float(latest["open_0930"]) if not pd.isna(latest["open_0930"]) else float("nan")
    base_qty = int(math.floor((float(account_equity) * leverage) / day_open)) if day_open > 0 else 0
    return leverage, day_open, base_qty


def _position_lookup(
    positions: list[Position] | None,
    *,
    symbol: str,
) -> Position | None:
    """Find the current position for one symbol."""
    for position in positions or []:
        if str(position.symbol).upper() == str(symbol).upper() and int(position.qty) > 0:
            return position
    return None


def _position_to_open_trade(position: Position, *, timestamp: pd.Timestamp) -> OpenTrade:
    """Project a live position into the legacy stop-helper trade shape."""
    return OpenTrade(
        side=int(position.side),
        shares=int(position.qty),
        entry_timestamp=pd.Timestamp(timestamp),
        entry_price=float(position.avg_price),
        entry_cost=0.0,
        decision_timestamp=pd.Timestamp(timestamp),
    )


def _stop_status(
    row: pd.Series,
    *,
    position: Position | None,
    variant: LiveStrategyVariant,
    is_decision_time: bool,
) -> tuple[bool, str]:
    """Evaluate whether a live position should be stopped now."""
    if position is None or int(position.qty) <= 0:
        return False, "none"

    trade = _position_to_open_trade(position, timestamp=pd.Timestamp(row["timestamp"]))
    if is_decision_time:
        return bool(_stop_triggered(trade, row)), "decision_stop"
    if not variant.minute_stop_monitoring:
        return False, "disabled"
    if variant.hybrid_stop_mode:
        hit, _ = _hybrid_stop_trigger_details(
            trade,
            row,
            catastrophic_stop_bps=variant.catastrophic_stop_bps,
        )
        return bool(hit), "hybrid_catastrophic"
    hit, _ = stop_trigger_details(trade, row, minute_aware=True)
    return bool(hit), "minute_full"


def compute_live_strategy_snapshot(
    enriched_bars: pd.DataFrame,
    *,
    variant: LiveStrategyVariant,
    account_equity: float = 100000.0,
    positions: list[Position] | None = None,
    artifact_bundle: Mapping[str, Any] | None = None,
    symbol: str = "SPY",
    sigma_target: float = 0.02,
    lev_cap: float = 4.0,
) -> dict[str, Any]:
    """Build one live strategy snapshot from the latest enriched bar."""
    if enriched_bars.empty:
        return {
            "strategy": variant.name,
            "symbol": str(symbol).upper(),
            "timestamp": pd.NaT,
            "signal": "no_data",
            "desired_side": 0,
            "actionable_now": False,
            "is_decision_time": False,
            "should_flatten": False,
            "base_qty": 0,
            "target_qty": 0,
            "p_good": float("nan"),
            "size_mult": 1.0,
            "overlay_enabled": True,
            "position_side": "flat",
            "position_qty": 0,
            "stop_triggered": False,
            "stop_mode": "none",
            "stream_state": "no_data",
            "variant_description": variant.description,
        }

    row = enriched_bars.iloc[-1]
    ts = pd.Timestamp(row["timestamp"])
    strategy = BaselineNoiseAreaStrategy(
        symbol=str(symbol).upper(),
        decision_freq_mins=variant.decision_freq_mins,
        first_trade_time=variant.first_trade_time,
    )
    signal = strategy.on_decision({"row": row, "symbol": symbol})
    desired_side = int(signal.desired_side)
    is_decision_time = bool(strategy.is_decision_time(ts))
    should_flatten = bool(strategy.should_flatten(ts))
    leverage, day_open, base_qty = _latest_daily_sizing(
        enriched_bars,
        account_equity=account_equity,
        sigma_target=sigma_target,
        lev_cap=lev_cap,
    )
    position = _position_lookup(positions, symbol=symbol)
    stop_triggered, stop_mode = _stop_status(
        row,
        position=position,
        variant=variant,
        is_decision_time=is_decision_time,
    )

    p_good = float("nan")
    size_mult = 1.0
    overlay_enabled = True
    target_qty = int(base_qty) if desired_side != 0 else 0
    if variant.use_ml and desired_side != 0:
        bundle = dict(artifact_bundle) if artifact_bundle is not None else _load_artifact_bundle(None, None, None)
        p_good = _safe_calibrated_probability(bundle, row, desired_side)
        sizer = MLOutputSizerRiskManager(
            threshold=float(bundle["threshold"]),
            prob_q20=float(bundle["prob_q20"]) if bundle["prob_q20"] is not None else None,
            prob_q40=float(bundle["prob_q40"]) if bundle["prob_q40"] is not None else None,
            prob_q60=float(bundle["prob_q60"]) if bundle["prob_q60"] is not None else None,
            prob_q80=float(bundle["prob_q80"]) if bundle["prob_q80"] is not None else None,
            allocation_mode="soft_size",
            neutral_zone=True,
            size_floor=variant.size_floor,
            size_cap=variant.size_cap,
            regime_overlay=variant.regime_overlay,
        )
        target_qty = sizer.size(
            Signal(timestamp=ts, symbol=str(symbol).upper(), desired_side=desired_side, confidence=p_good),
            account={"equity": float(account_equity)},
            market_state={
                "base_qty": int(base_qty),
                "p_good": p_good,
                "timestamp": ts,
                "row": row,
                "closed_trade_history": [],
            },
        )
        size_mult = float(sizer.last_details.get("size_mult", 1.0))
        overlay_enabled = bool(sizer.last_details.get("overlay_enabled", True))

    signal_label = {1: "long", -1: "short", 0: "flat"}[desired_side]
    position_side = "flat"
    position_qty = 0
    if position is not None:
        position_side = "long" if int(position.side) > 0 else "short"
        position_qty = int(position.qty)

    return {
        "strategy": variant.name,
        "symbol": str(symbol).upper(),
        "timestamp": ts,
        "signal": signal_label,
        "desired_side": desired_side,
        "actionable_now": bool(is_decision_time or should_flatten or stop_triggered),
        "is_decision_time": is_decision_time,
        "should_flatten": should_flatten,
        "close": float(row["close"]),
        "UB": float(row["UB"]),
        "LB": float(row["LB"]),
        "VWAP": float(row["VWAP"]),
        "breakout_margin_bps": float(abs(float(row["close"] - row["UB"])) / float(row["close"]) * 10000.0)
        if desired_side > 0 and float(row["close"]) != 0.0
        else float(abs(float(row["LB"] - row["close"])) / float(row["close"]) * 10000.0)
        if desired_side < 0 and float(row["close"]) != 0.0
        else 0.0,
        "account_equity": float(account_equity),
        "leverage": float(leverage),
        "day_open": float(day_open) if not pd.isna(day_open) else float("nan"),
        "base_qty": int(base_qty),
        "target_qty": int(target_qty),
        "p_good": float(p_good) if not pd.isna(p_good) else float("nan"),
        "size_mult": float(size_mult),
        "overlay_enabled": bool(overlay_enabled),
        "position_side": position_side,
        "position_qty": int(position_qty),
        "stop_triggered": bool(stop_triggered),
        "stop_mode": stop_mode,
        "stream_state": "live",
        "hybrid_stop_mode": bool(variant.hybrid_stop_mode),
        "catastrophic_stop_bps": float(variant.catastrophic_stop_bps),
        "variant_description": variant.description,
    }


def compute_live_strategy_board(
    enriched_bars: pd.DataFrame,
    *,
    strategy_names: list[str] | None = None,
    account_equity: float = 100000.0,
    positions: list[Position] | None = None,
    symbol: str = "SPY",
    sigma_target: float = 0.02,
    lev_cap: float = 4.0,
    artifact_bundle: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate the current live state for baseline and the selected variants."""
    names = strategy_names or list_live_variants()
    rows = [
        compute_live_strategy_snapshot(
            enriched_bars,
            variant=get_live_variant(name),
            account_equity=account_equity,
            positions=positions,
            artifact_bundle=artifact_bundle if get_live_variant(name).use_ml else None,
            symbol=symbol,
            sigma_target=sigma_target,
            lev_cap=lev_cap,
        )
        for name in names
    ]
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    sort_cols = ["actionable_now", "desired_side", "target_qty"]
    return board.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)


def save_live_strategy_board(board: pd.DataFrame, *, output_dir: Path | None = None) -> dict[str, Path]:
    """Persist the latest strategy board for inspection and deployment proof."""
    config = load_config()
    live_dir = (output_dir or (config.data_dir / "live")).expanduser()
    live_dir.mkdir(parents=True, exist_ok=True)
    csv_path = live_dir / "live_strategy_board_latest.csv"
    json_path = live_dir / "live_strategy_board_latest.json"
    board.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(board.to_dict(orient="records"), f, indent=2, default=str)
    return {"csv": csv_path, "json": json_path}


class LivePaperStrategyRunner:
    """Simple paper-trading loop for one selected live variant.

    The runtime intentionally stays conservative:

    - it uses Alpaca paper routing only;
    - it routes one strategy at a time;
    - it does not modify the research/backtest state or outputs.
    """

    def __init__(
        self,
        *,
        market_data: AlpacaLiveMarketData,
        broker: AlpacaPaperBroker,
        variant_name: str,
        symbol: str = "SPY",
        sigma_target: float = 0.02,
        lev_cap: float = 4.0,
        artifact_bundle: Mapping[str, Any] | None = None,
        dry_run: bool = False,
        min_live_timestamp: pd.Timestamp | None = None,
    ) -> None:
        self.market_data = market_data
        self.broker = broker
        self.variant = get_live_variant(variant_name)
        self.symbol = str(symbol).upper()
        self.sigma_target = float(sigma_target)
        self.lev_cap = float(lev_cap)
        self.artifact_bundle = dict(artifact_bundle) if artifact_bundle is not None else None
        self.dry_run = bool(dry_run)
        self._last_processed_ts: pd.Timestamp | None = None
        self.min_live_timestamp = pd.Timestamp(min_live_timestamp) if min_live_timestamp is not None else None
        self._warmup_emitted = False

    def step(self) -> dict[str, Any] | None:
        """Process the latest bar once and optionally route paper orders."""
        enriched = self.market_data.enriched_bars()
        if enriched.empty:
            return None
        latest_ts = pd.Timestamp(enriched["timestamp"].iloc[-1])
        if self.min_live_timestamp is not None and latest_ts <= self.min_live_timestamp:
            if self._warmup_emitted:
                return None
            self._warmup_emitted = True
            snapshot = compute_live_strategy_snapshot(
                enriched,
                variant=self.variant,
                account_equity=_latest_account_equity(self.broker.get_account(), 100000.0),
                positions=self.broker.get_positions(),
                artifact_bundle=self.artifact_bundle,
                symbol=self.symbol,
                sigma_target=self.sigma_target,
                lev_cap=self.lev_cap,
            )
            snapshot["actionable_now"] = False
            snapshot["runtime_action"] = "warming_live_stream"
            snapshot["runtime_order_qty"] = 0
            snapshot["dry_run"] = bool(self.dry_run)
            snapshot["stream_state"] = "warming_live_stream"
            return snapshot
        if self._last_processed_ts is not None and latest_ts <= self._last_processed_ts:
            return None
        self._last_processed_ts = latest_ts

        positions = self.broker.get_positions()
        account = self.broker.get_account()
        equity = _latest_account_equity(account, 100000.0)
        snapshot = compute_live_strategy_snapshot(
            enriched,
            variant=self.variant,
            account_equity=equity,
            positions=positions,
            artifact_bundle=self.artifact_bundle,
            symbol=self.symbol,
            sigma_target=self.sigma_target,
            lev_cap=self.lev_cap,
        )
        current_position = _position_lookup(positions, symbol=self.symbol)
        current_side = Side.FLAT if current_position is None else Side.from_value(current_position.side)
        desired_side = Side.from_value(snapshot["desired_side"])
        row = enriched.iloc[-1]
        strategy = BaselineNoiseAreaStrategy(
            symbol=self.symbol,
            decision_freq_mins=self.variant.decision_freq_mins,
            first_trade_time=self.variant.first_trade_time,
        )

        action = "hold"
        order_qty = 0

        if snapshot["stop_triggered"] and current_position is not None:
            action = f"flatten_on_{snapshot['stop_mode']}"
            order_qty = int(current_position.qty)
            if not self.dry_run:
                self.broker.flatten_symbol(self.symbol)
        elif bool(snapshot["should_flatten"]) and current_position is not None:
            action = "flatten_eod"
            order_qty = int(current_position.qty)
            if not self.dry_run:
                self.broker.flatten_symbol(self.symbol)
        elif bool(snapshot["is_decision_time"]):
            if current_position is None and desired_side != Side.FLAT:
                if strategy.allow_open(latest_ts, current_side, desired_side, row=row):
                    action = "open"
                    order_qty = int(snapshot["target_qty"])
                    if order_qty > 0 and not self.dry_run:
                        self.broker.submit_market_order(self.symbol, desired_side, order_qty)
            elif current_position is not None and desired_side == Side(-int(current_side)):
                if strategy.allow_open(latest_ts, current_side, desired_side, row=row):
                    action = "flip"
                    order_qty = int(snapshot["target_qty"])
                    if not self.dry_run:
                        self.broker.flatten_symbol(self.symbol)
                        if order_qty > 0:
                            self.broker.submit_market_order(self.symbol, desired_side, order_qty)
                else:
                    action = "flip_blocked_hold"

        snapshot["runtime_action"] = action
        snapshot["runtime_order_qty"] = int(order_qty)
        snapshot["dry_run"] = bool(self.dry_run)
        snapshot["stream_state"] = "live"
        return snapshot

    def run(
        self,
        *,
        refresh_seconds: float = 5.0,
        duration_seconds: float | None = None,
    ) -> pd.DataFrame:
        """Run the live paper loop and return the captured decision snapshots."""
        if not self.market_data.is_running:
            self.market_data.start()
        if not self.broker.is_streaming:
            self.broker.start_trade_updates()

        deadline = None if duration_seconds is None else time.time() + float(duration_seconds)
        rows: list[dict[str, Any]] = []
        while deadline is None or time.time() < deadline:
            snapshot = self.step()
            if snapshot is not None:
                rows.append(snapshot)
                logger.info(
                    "live strategy=%s ts=%s signal=%s action=%s qty=%s p_good=%s",
                    snapshot["strategy"],
                    snapshot["timestamp"],
                    snapshot["signal"],
                    snapshot["runtime_action"],
                    snapshot["runtime_order_qty"],
                    snapshot["p_good"],
                )
            time.sleep(float(refresh_seconds))
        return pd.DataFrame(rows)


def run_live_strategy_board_loop(
    *,
    market_data: AlpacaLiveMarketData,
    broker: AlpacaPaperBroker | None = None,
    strategy_names: list[str] | None = None,
    symbol: str = "SPY",
    refresh_seconds: float = 5.0,
    duration_seconds: float | None = 300.0,
    output_dir: Path | None = None,
    min_live_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Monitor the live strategy board over websocket bars and persist snapshots."""
    if not market_data.is_running:
        market_data.start()
    if broker is not None and not broker.is_streaming:
        broker.start_trade_updates()

    rows: list[pd.DataFrame] = []
    deadline = None if duration_seconds is None else time.time() + float(duration_seconds)
    warmup_logged = False
    while deadline is None or time.time() < deadline:
        enriched = market_data.enriched_bars()
        latest_ts = _latest_buffer_timestamp(enriched)
        positions = broker.get_positions() if broker is not None else []
        account = broker.get_account() if broker is not None else {}
        equity = _latest_account_equity(account, 100000.0)
        board = compute_live_strategy_board(
            enriched,
            strategy_names=strategy_names,
            account_equity=equity,
            positions=positions,
            symbol=symbol,
        )
        if not board.empty:
            if min_live_timestamp is not None and latest_ts is not None and latest_ts <= pd.Timestamp(min_live_timestamp):
                if warmup_logged:
                    time.sleep(float(refresh_seconds))
                    continue
                warmup_logged = True
                board = board.copy()
                board["actionable_now"] = False
                board["stream_state"] = "warming_live_stream"
            else:
                board = board.copy()
                board["stream_state"] = "live"
            rows.append(board.assign(snapshot_ts=pd.Timestamp.utcnow()))
            save_live_strategy_board(board, output_dir=output_dir)
            logger.info("\n%s", board.to_string(index=False))
        time.sleep(float(refresh_seconds))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
