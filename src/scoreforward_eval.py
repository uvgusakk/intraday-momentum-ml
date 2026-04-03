"""True rolling retrain / score-forward evaluation for the serious ML methods.

This module keeps the public API intentionally small:

- ``ScoreforwardConfig`` defines the backtest and split parameters.
- ``run_ml_scoreforward_backtests(...)`` is the stable function used by the CLI.

Internally the work now lives in ``ScoreforwardRunner`` so the sequence is
explicit and readable:

1. normalize/enrich the bars needed by the baseline and ML paths;
2. build the ML candidate dataset once;
3. build chronological train/validation/test splits;
4. retrain on each split and score only the next held-out window;
5. chain AUM forward across windows and summarize results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .backtest_ml_filter import run_ml_filtered_backtest
from .baseline_strategy import _compute_daily_sizing_table, _normalize_df, run_baseline_backtest
from .config import (
    DEFAULT_DECISION_FREQ_MINS,
    DEFAULT_EXECUTION_SPREAD_BPS,
    DEFAULT_ML_LABEL_HORIZON_MINS,
    DEFAULT_MINUTE_STOP_MONITORING,
    DEFAULT_MM_CAP,
    DEFAULT_MM_FLOOR,
    DEFAULT_MM_LOOKBACK_DAYS,
    DEFAULT_SCOREFORWARD_LABEL_MODE,
    DEFAULT_SOFT_SIZE_CAP,
    DEFAULT_SOFT_SIZE_FLOOR,
    DEFAULT_USE_NEXT_BAR_OPEN,
)
from .features_ml import build_ml_dataset
from .train_ml import fit_best_model_bundle, make_split_specs


@dataclass(frozen=True)
class SeriousMethodSpec:
    """Execution settings for a single serious ML method.

    The strategy direction always comes from the baseline signal. These method
    specs only change how the ML overlay sizes the already-approved baseline
    trades.
    """

    name: str
    backtest_kwargs: dict[str, Any]


SERIOUS_METHODS: dict[str, SeriousMethodSpec] = {
    "soft": SeriousMethodSpec(
        name="soft",
        backtest_kwargs={
            "regime_overlay": False,
            "size_floor": DEFAULT_SOFT_SIZE_FLOOR,
            "size_cap": DEFAULT_SOFT_SIZE_CAP,
        },
    ),
    "mm": SeriousMethodSpec(
        name="mm",
        backtest_kwargs={
            "size_floor": DEFAULT_SOFT_SIZE_FLOOR,
            "size_cap": DEFAULT_SOFT_SIZE_CAP,
            "market_vol_overlay": True,
            "market_vol_lookback_days": DEFAULT_MM_LOOKBACK_DAYS,
            "market_vol_floor": DEFAULT_MM_FLOOR,
            "market_vol_cap": DEFAULT_MM_CAP,
            "regime_overlay": False,
        },
    ),
}


@dataclass(frozen=True)
class ScoreforwardConfig:
    """Parameters for the corrected score-forward evaluation."""

    initial_aum: float = 100000.0
    sigma_target: float = 0.02
    lev_cap: float = 4.0
    commission_per_share: float = 0.0035
    slippage_per_share: float = 0.001
    decision_freq_mins: int = DEFAULT_DECISION_FREQ_MINS
    first_trade_time: str = "10:00"
    use_next_bar_open: bool = DEFAULT_USE_NEXT_BAR_OPEN
    minute_stop_monitoring: bool = DEFAULT_MINUTE_STOP_MONITORING
    spread_bps: float = DEFAULT_EXECUTION_SPREAD_BPS
    label_mode: str = DEFAULT_SCOREFORWARD_LABEL_MODE
    horizon_mins: int = DEFAULT_ML_LABEL_HORIZON_MINS
    train_months: int = 24
    val_months: int = 3
    test_months: int = 3
    step_months: int = 3
    warmup_days: int = 90


@dataclass
class MethodState:
    """Mutable state tracked while chaining split-level results forward."""

    current_aum: float
    equity_frames: list[pd.DataFrame] = field(default_factory=list)
    trade_frames: list[pd.DataFrame] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedData:
    """Normalized inputs shared across all split executions."""

    bars_norm: pd.DataFrame
    unique_days: list[pd.Timestamp]
    full_sizing: pd.DataFrame
    ml_df: pd.DataFrame
    split_specs: list[dict[str, Any]]


def _prepare_ml_dataframe(bars: pd.DataFrame, *, config: ScoreforwardConfig) -> pd.DataFrame:
    """Build the ML candidate dataset used for split creation and training."""
    X, y, meta = build_ml_dataset(
        bars,
        backtest_kwargs={
            "initial_aum": config.initial_aum,
            "sigma_target": config.sigma_target,
            "lev_cap": config.lev_cap,
            "commission_per_share": config.commission_per_share,
            "slippage_per_share": config.slippage_per_share,
            "decision_freq_mins": config.decision_freq_mins,
            "first_trade_time": config.first_trade_time,
            "use_next_bar_open": config.use_next_bar_open,
            "minute_stop_monitoring": config.minute_stop_monitoring,
            "spread_bps": config.spread_bps,
        },
        label_mode=config.label_mode,
        horizon_mins=config.horizon_mins,
        persist=False,
    )
    df = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    if df.empty:
        raise ValueError("ML dataset is empty; cannot run score-forward evaluation")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["date", "timestamp"]).reset_index(drop=True)


def _normalize_day_series(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Normalize bars and extract the ordered trading-day index."""
    out = _normalize_df(bars)
    out["date_dt"] = pd.to_datetime(out["date"]).dt.normalize()
    days = list(pd.Index(out["date_dt"]).drop_duplicates().sort_values())
    return out, days


def _context_bars_for_test(
    bars: pd.DataFrame,
    unique_days: list[pd.Timestamp],
    test_days: list[pd.Timestamp],
    warmup_days: int,
) -> pd.DataFrame:
    """Keep the test window plus enough prior history for intraday indicators."""
    first_test = min(test_days)
    last_test = max(test_days)
    first_idx = unique_days.index(first_test)
    last_idx = unique_days.index(last_test)
    start_idx = max(0, first_idx - int(warmup_days))
    keep_days = set(unique_days[start_idx:last_idx + 1])
    return bars.loc[bars["date_dt"].isin(keep_days)].drop(columns=["date_dt"]).copy()


def _test_days_from_split(split: dict[str, Any], bars: pd.DataFrame) -> list[pd.Timestamp]:
    """Translate a train_ml split specification into explicit daily test dates."""
    date_dt = pd.to_datetime(bars["date_dt"]).dt.normalize()
    if split.get("split_type") == "walk_forward_monthly":
        test_periods = {pd.Period(x, freq="M") for x in split["test_label"]}
        mask = date_dt.dt.to_period("M").isin(test_periods)
        return sorted(date_dt.loc[mask].drop_duplicates().tolist())
    if split.get("split_type") == "chrono_day_fallback":
        start = pd.Timestamp(split["test_label"][0]).normalize()
        end = pd.Timestamp(split["test_label"][-1]).normalize()
        mask = (date_dt >= start) & (date_dt <= end)
        return sorted(date_dt.loc[mask].drop_duplicates().tolist())
    raise ValueError(f"Unsupported split_type: {split.get('split_type')}")


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty dataframes while keeping empty-output behavior simple."""
    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def _combine_split_metrics(
    equity_curve: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    initial_aum: float,
) -> dict[str, float]:
    """Summarize the chained score-forward path for one method."""
    if equity_curve.empty:
        return {
            "final_equity": float(initial_aum),
            "total_return_pct": 0.0,
            "trades_count": 0,
            "turnover": 0.0,
            "total_costs": 0.0,
        }

    from .metrics import summarize_backtest

    summary = summarize_backtest(equity_curve["daily_return"])
    mean_equity = float(equity_curve["equity"].mean()) if not equity_curve.empty else 0.0
    total_notional = (
        float(
            trades_df["entry_price"].abs().mul(trades_df["shares"]).sum()
            + trades_df["exit_price"].abs().mul(trades_df["shares"]).sum()
        )
        if not trades_df.empty
        else 0.0
    )
    summary.update(
        {
            "final_equity": float(equity_curve["equity"].iloc[-1]),
            "total_return_pct": float((equity_curve["equity"].iloc[-1] / initial_aum - 1.0) * 100.0),
            "trades_count": int(len(trades_df)),
            "turnover": float(total_notional / mean_equity) if mean_equity > 0 else 0.0,
            "total_costs": float(trades_df["costs"].sum()) if not trades_df.empty else 0.0,
        }
    )
    return summary


class ScoreforwardRunner:
    """Object-oriented implementation of the corrected serious-method evaluation.

    The public function below is still the preferred entry point. This runner is
    here to keep the internal phases explicit and testable without changing the
    external output format.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        *,
        config: ScoreforwardConfig | None = None,
        methods: list[str] | None = None,
        method_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.bars = bars
        self.config = config or ScoreforwardConfig()
        self.methods = methods or ["soft", "mm"]
        self.method_overrides = method_overrides or {}
        invalid = [method for method in self.methods if method not in SERIOUS_METHODS]
        if invalid:
            raise ValueError(f"Unsupported serious methods: {invalid}. Allowed: {sorted(SERIOUS_METHODS)}")

    def run(self) -> dict[str, Any]:
        """Execute the full score-forward workflow and return the legacy payload."""
        prepared = self._prepare_data()
        method_states = self._initialize_method_states()
        split_rows: list[dict[str, Any]] = []

        for split in prepared.split_specs:
            self._run_split(
                split=split,
                prepared=prepared,
                method_states=method_states,
                split_rows=split_rows,
            )

        return self._build_output(method_states=method_states, split_rows=split_rows)

    def _prepare_data(self) -> PreparedData:
        """Create the shared inputs used by every split."""
        bars_norm, unique_days = _normalize_day_series(self.bars)
        full_sizing = _compute_daily_sizing_table(
            bars_norm,
            self.config.initial_aum,
            self.config.sigma_target,
            self.config.lev_cap,
        )
        ml_df = _prepare_ml_dataframe(bars_norm, config=self.config)
        split_specs = make_split_specs(
            ml_df,
            train_months=self.config.train_months,
            val_months=self.config.val_months,
            test_months=self.config.test_months,
            step_months=self.config.step_months,
        )
        if not split_specs:
            raise ValueError("No valid score-forward splits were created")

        return PreparedData(
            bars_norm=bars_norm,
            unique_days=unique_days,
            full_sizing=full_sizing,
            ml_df=ml_df,
            split_specs=split_specs,
        )

    def _initialize_method_states(self) -> dict[str, MethodState]:
        """Initialize chained-AUM state for baseline and the requested methods."""
        states = {"baseline": MethodState(current_aum=float(self.config.initial_aum))}
        for method in self.methods:
            states[method] = MethodState(current_aum=float(self.config.initial_aum))
        return states

    def _run_split(
        self,
        *,
        split: dict[str, Any],
        prepared: PreparedData,
        method_states: dict[str, MethodState],
        split_rows: list[dict[str, Any]],
    ) -> None:
        """Run one chronological split for baseline plus every serious ML method."""
        train_df = prepared.ml_df.loc[split["train_mask"]].copy()
        val_df = prepared.ml_df.loc[split["val_mask"]].copy()
        test_df = prepared.ml_df.loc[split["test_mask"]].copy()
        if train_df.empty or val_df.empty or test_df.empty:
            return

        bundle_info = fit_best_model_bundle(train_df, val_df, test_df)
        artifact_bundle = dict(bundle_info["best_bundle"])

        test_days = _test_days_from_split(split, prepared.bars_norm)
        context_bars = _context_bars_for_test(
            prepared.bars_norm,
            prepared.unique_days,
            test_days,
            self.config.warmup_days,
        )
        first_test_day = pd.Timestamp(test_days[0]).strftime("%Y-%m-%d")
        # Baseline should trade only on the held-out days for fair comparison.
        live_bars = context_bars.loc[
            pd.to_datetime(context_bars["date"]).dt.normalize().isin(test_days)
        ].copy()

        self._run_baseline_split(
            split_id=int(split["split_id"]),
            live_bars=live_bars,
            full_sizing=prepared.full_sizing,
            method_state=method_states["baseline"],
            split_rows=split_rows,
        )

        for method in self.methods:
            self._run_ml_split(
                method=method,
                split_id=int(split["split_id"]),
                context_bars=context_bars,
                first_test_day=first_test_day,
                full_sizing=prepared.full_sizing,
                artifact_bundle=artifact_bundle,
                method_state=method_states[method],
                split_rows=split_rows,
            )

    def _run_baseline_split(
        self,
        *,
        split_id: int,
        live_bars: pd.DataFrame,
        full_sizing: pd.DataFrame,
        method_state: MethodState,
        split_rows: list[dict[str, Any]],
    ) -> None:
        """Execute the paper baseline on the current held-out window."""
        baseline_out = run_baseline_backtest(
            live_bars,
            initial_aum=method_state.current_aum,
            sigma_target=self.config.sigma_target,
            lev_cap=self.config.lev_cap,
            commission_per_share=self.config.commission_per_share,
            slippage_per_share=self.config.slippage_per_share,
            decision_freq_mins=self.config.decision_freq_mins,
            first_trade_time=self.config.first_trade_time,
            use_next_bar_open=self.config.use_next_bar_open,
            minute_stop_monitoring=self.config.minute_stop_monitoring,
            spread_bps=self.config.spread_bps,
            daily_sizing=full_sizing,
        )
        method_state.equity_frames.append(baseline_out["equity_curve"])
        method_state.trade_frames.append(baseline_out["trades"])
        method_state.current_aum = float(baseline_out["summary"]["final_equity"])
        split_rows.append(
            {
                "split_id": split_id,
                "method": "baseline",
                **baseline_out["summary"],
            }
        )

    def _run_ml_split(
        self,
        *,
        method: str,
        split_id: int,
        context_bars: pd.DataFrame,
        first_test_day: str,
        full_sizing: pd.DataFrame,
        artifact_bundle: dict[str, Any],
        method_state: MethodState,
        split_rows: list[dict[str, Any]],
    ) -> None:
        """Execute one serious ML sizing method on the held-out window."""
        method_kwargs = dict(SERIOUS_METHODS[method].backtest_kwargs)
        method_kwargs.update(self.method_overrides.get(method, {}))

        out = run_ml_filtered_backtest(
            context_bars,
            initial_aum=method_state.current_aum,
            sigma_target=self.config.sigma_target,
            lev_cap=self.config.lev_cap,
            commission_per_share=self.config.commission_per_share,
            slippage_per_share=self.config.slippage_per_share,
            decision_freq_mins=self.config.decision_freq_mins,
            first_trade_time=self.config.first_trade_time,
            use_next_bar_open=self.config.use_next_bar_open,
            minute_stop_monitoring=self.config.minute_stop_monitoring,
            spread_bps=self.config.spread_bps,
            artifact_bundle=artifact_bundle,
            daily_sizing=full_sizing,
            trade_start_date=first_test_day,
            allocation_mode="soft_size",
            **method_kwargs,
        )
        method_state.equity_frames.append(out["equity_curve"])
        method_state.trade_frames.append(out["trades"])
        method_state.current_aum = float(out["metrics"]["final_equity"])
        split_rows.append(
            {
                "split_id": split_id,
                "method": method,
                **out["metrics"],
            }
        )

    def _build_output(
        self,
        *,
        method_states: dict[str, MethodState],
        split_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the legacy output payload expected by the CLI and notebook."""
        summary_rows: list[dict[str, Any]] = []
        combined_outputs: dict[str, dict[str, Any]] = {}

        for method, state in method_states.items():
            equity_curve = _concat_frames(state.equity_frames)
            trades_df = _concat_frames(state.trade_frames)
            summary = _combine_split_metrics(
                equity_curve,
                trades_df,
                initial_aum=self.config.initial_aum,
            )
            summary_rows.append({"method": method, **summary})
            combined_outputs[method] = {
                "equity_curve": equity_curve,
                "trades": trades_df,
                "summary": summary,
            }

        split_df = pd.DataFrame(split_rows)
        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values(["sharpe", "final_equity"], ascending=[False, False])
            .reset_index(drop=True)
        )
        return {
            "split_metrics": split_df,
            "summary": summary_df,
            "outputs": combined_outputs,
        }


def run_ml_scoreforward_backtests(
    bars: pd.DataFrame,
    *,
    config: ScoreforwardConfig | None = None,
    methods: list[str] | None = None,
    method_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stable public wrapper used by the CLI and notebook code."""
    runner = ScoreforwardRunner(
        bars,
        config=config,
        methods=methods,
        method_overrides=method_overrides,
    )
    return runner.run()
