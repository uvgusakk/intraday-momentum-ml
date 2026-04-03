"""Walk-forward training utilities for trade-filter classifiers.

This module now supports two target regimes:

- ``binary``: the legacy positive-vs-negative target already stored in the
  dataset.
- ``large_winner``: a split-local target derived from train-window return
  quantiles. This avoids leaking a global threshold from future data into the
  training labels.

It also supports side-filtered training (all/long/short) so notebooks can train
separate models without re-implementing the walk-forward machinery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config

logger = logging.getLogger(__name__)


@dataclass
class SplitDef:
    """Date-based walk-forward split definition."""

    split_id: int
    train_months: list[pd.Period]
    val_months: list[pd.Period]
    test_months: list[pd.Period]


class PlattCalibrator:
    """Simple Platt scaling calibrator using logistic regression on model scores."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(max_iter=1000)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(scores.reshape(-1, 1), y)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _load_ml_dataset() -> pd.DataFrame:
    config = load_config()
    path = config.data_dir / "ml_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)
    required = {"date", "timestamp", "side", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ml_dataset.parquet missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["date", "timestamp"]).reset_index(drop=True)


def _coerce_side_filter(df: pd.DataFrame, side_filter: str) -> pd.DataFrame:
    if side_filter not in {"all", "long", "short"}:
        raise ValueError("side_filter must be one of: 'all', 'long', 'short'")
    if side_filter == "all":
        return df.copy()
    side_value = 1 if side_filter == "long" else -1
    return df.loc[pd.to_numeric(df["side"], errors="coerce") == side_value].copy()


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"date", "timestamp", "side", "symbol", "y", "pnl", "trade_return", "net_pnl", "return", "costs"}
    cols = [c for c in df.columns if c not in excluded]
    if not cols:
        raise ValueError("No feature columns found in dataset.")
    return cols


def _trade_return_series(df: pd.DataFrame) -> pd.Series:
    for c in ["trade_return", "pnl", "net_pnl", "return"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Fallback proxy if only labels are available.
    return df["y"].map({1: 1.0, 0: -1.0}).astype(float)


def _build_target_series(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_mode: str,
    target_quantile: float,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, float]]:
    if target_mode not in {"binary", "large_winner"}:
        raise ValueError("target_mode must be one of: 'binary', 'large_winner'")

    if target_mode == "binary":
        return (
            train_df["y"].astype(int),
            val_df["y"].astype(int),
            test_df["y"].astype(int),
            {"target_mode": "binary", "target_quantile": float("nan"), "train_large_winner_cut": float("nan")},
        )

    if not 0.5 < float(target_quantile) < 1.0:
        raise ValueError("target_quantile must be in (0.5, 1.0) for large_winner mode")

    train_ret = _trade_return_series(train_df)
    val_ret = _trade_return_series(val_df)
    test_ret = _trade_return_series(test_df)
    cut = float(np.quantile(train_ret, float(target_quantile)))

    y_train = (train_ret >= cut).astype(int)
    y_val = (val_ret >= cut).astype(int)
    y_test = (test_ret >= cut).astype(int)
    meta = {
        "target_mode": "large_winner",
        "target_quantile": float(target_quantile),
        "train_large_winner_cut": cut,
    }
    return y_train, y_val, y_test, meta


def _make_walk_forward_splits(
    df: pd.DataFrame,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
) -> list[SplitDef]:
    months = sorted(df["date"].dt.to_period("M").unique())
    needed = train_months + val_months + test_months
    if len(months) < needed:
        raise ValueError(
            f"Not enough history for walk-forward splits: need {needed} months, got {len(months)}"
        )

    splits: list[SplitDef] = []
    sid = 0
    i = 0
    while i + needed <= len(months):
        tr = months[i : i + train_months]
        va = months[i + train_months : i + train_months + val_months]
        te = months[
            i + train_months + val_months : i + train_months + val_months + test_months
        ]
        splits.append(SplitDef(split_id=sid, train_months=tr, val_months=va, test_months=te))
        sid += 1
        i += step_months
    return splits


def _make_chrono_day_split(df: pd.DataFrame) -> dict[str, Any]:
    """Build one chronological train/val/test split by trading day."""
    unique_days = sorted(df["date"].dt.normalize().unique())
    n = len(unique_days)
    if n < 3:
        raise ValueError(
            f"Not enough unique trading days for fallback split: need >=3, got {n}"
        )

    train_end = max(1, int(n * 0.6))
    val_end = max(train_end + 1, int(n * 0.8))
    val_end = min(val_end, n - 1)

    train_days = unique_days[:train_end]
    val_days = unique_days[train_end:val_end]
    test_days = unique_days[val_end:]

    if len(val_days) == 0 or len(test_days) == 0:
        raise ValueError(
            "Fallback split failed to allocate validation/test days. "
            "Increase fetched date range."
        )

    return {
        "split_id": 0,
        "split_type": "chrono_day_fallback",
        "train_mask": df["date"].isin(train_days),
        "val_mask": df["date"].isin(val_days),
        "test_mask": df["date"].isin(test_days),
        "train_label": [str(pd.Timestamp(train_days[0]).date()), str(pd.Timestamp(train_days[-1]).date())],
        "val_label": [str(pd.Timestamp(val_days[0]).date()), str(pd.Timestamp(val_days[-1]).date())],
        "test_label": [str(pd.Timestamp(test_days[0]).date()), str(pd.Timestamp(test_days[-1]).date())],
    }


def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> Any:
    """Train a logistic regression classifier with standardization."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def train_lightgbm_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Any:
    """Train a compact LightGBM classifier with optional early stopping."""
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )

    fit_kwargs: dict[str, Any] = {}
    if X_val is not None and y_val is not None and len(X_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "auc"
        fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=50, verbose=False)]

    model.fit(X, y, **fit_kwargs)
    return model


def _raw_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    proba = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    eps = 1e-8
    proba = np.clip(proba, eps, 1 - eps)
    return np.log(proba / (1 - proba))


def _classification_metrics(y_true: pd.Series, p: np.ndarray, thr: float) -> dict[str, float]:
    y_hat = (p >= thr).astype(int)
    try:
        auc = float(roc_auc_score(y_true, p))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
    }


def _trading_metrics(trade_returns: pd.Series, p: np.ndarray, thr: float) -> dict[str, float]:
    mask = p >= thr
    taken = pd.Series(trade_returns).loc[mask]
    n = int(mask.sum())
    if n == 0:
        return {
            "n_trades": 0.0,
            "take_rate": 0.0,
            "net_mean_return": 0.0,
            "net_sharpe": -999.0,
        }

    mu = float(taken.mean())
    sigma = float(taken.std(ddof=0))
    sharpe = 0.0 if sigma == 0 else (mu / sigma) * np.sqrt(252.0)
    return {
        "n_trades": float(n),
        "take_rate": float(n / len(mask)),
        "net_mean_return": mu,
        "net_sharpe": float(sharpe),
    }


def _select_threshold(y_val: pd.Series, p_val: np.ndarray, ret_val: pd.Series) -> tuple[float, dict[str, float]]:
    best_thr = 0.5
    best_metrics = _trading_metrics(ret_val, p_val, best_thr)

    for thr in np.arange(0.50, 0.96, 0.02):
        m = _trading_metrics(ret_val, p_val, float(thr))
        better = (m["net_sharpe"] > best_metrics["net_sharpe"]) or (
            m["net_sharpe"] == best_metrics["net_sharpe"]
            and m["net_mean_return"] > best_metrics["net_mean_return"]
        )
        if better:
            best_thr = float(thr)
            best_metrics = m

    # Merge in classification stats at chosen threshold.
    best_metrics.update(_classification_metrics(y_val, p_val, best_thr))
    return best_thr, best_metrics


def make_split_specs(
    df: pd.DataFrame,
    *,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
) -> list[dict[str, Any]]:
    """Build chronological train/validation/test split specs.

    This helper is reused by both artifact training and score-forward
    evaluation. It returns non-overlapping test windows when `step_months`
    matches `test_months`, which is the intended evaluation mode for the
    project.
    """
    months_available = len(df["date"].dt.to_period("M").unique())
    needed = train_months + val_months + test_months

    split_specs: list[dict[str, Any]] = []
    if months_available >= needed:
        for split in _make_walk_forward_splits(
            df,
            train_months=train_months,
            val_months=val_months,
            test_months=test_months,
            step_months=step_months,
        ):
            month_series = df["date"].dt.to_period("M")
            split_specs.append(
                {
                    "split_id": split.split_id,
                    "split_type": "walk_forward_monthly",
                    "train_mask": month_series.isin(split.train_months),
                    "val_mask": month_series.isin(split.val_months),
                    "test_mask": month_series.isin(split.test_months),
                    "train_label": [str(m) for m in split.train_months],
                    "val_label": [str(m) for m in split.val_months],
                    "test_label": [str(m) for m in split.test_months],
                }
            )
    else:
        logger.warning(
            "Insufficient monthly history for walk-forward (%s months available, %s required). "
            "Using one chronological day-based fallback split.",
            months_available,
            needed,
        )
        split_specs.append(_make_chrono_day_split(df))
    return split_specs


def fit_best_model_bundle(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_mode: str = "binary",
    target_quantile: float = 0.70,
) -> dict[str, Any]:
    """Fit candidate models on one split and return the best validated bundle.

    The selected bundle is the same object later persisted to disk by the
    artifact-training path, but returned in-memory so score-forward evaluation
    can retrain on each split without saving and reloading intermediate files.
    """
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train_df, val_df, and test_df must all be non-empty")

    X_cols = _feature_columns(train_df)
    y_train, y_val, y_test, target_meta = _build_target_series(
        train_df,
        val_df,
        test_df,
        target_mode=target_mode,
        target_quantile=target_quantile,
    )
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError("Need at least two classes in train and validation splits")

    X_train = train_df[X_cols]
    X_val = val_df[X_cols]
    X_test = test_df[X_cols]

    ret_val = _trade_return_series(val_df)
    ret_test = _trade_return_series(test_df)

    models = {
        "logistic": train_logistic_regression(X_train, y_train),
        "lightgbm": train_lightgbm_classifier(X_train, y_train, X_val=X_val, y_val=y_val),
    }

    split_models: dict[str, Any] = {}
    best_bundle: dict[str, Any] | None = None
    best_score = -1e18

    for model_name, model in models.items():
        val_scores = _raw_scores(model, X_val)
        calibrator = PlattCalibrator().fit(val_scores, y_val.to_numpy())

        p_val = calibrator.predict_proba(val_scores)
        thr, val_metrics = _select_threshold(y_val, p_val, ret_val)

        test_scores = _raw_scores(model, X_test)
        p_test = calibrator.predict_proba(test_scores)
        test_metrics = _classification_metrics(y_test, p_test, thr)
        test_metrics.update(_trading_metrics(ret_test, p_test, thr))

        split_models[model_name] = {
            "threshold": thr,
            "validation": val_metrics,
            "test": test_metrics,
        }

        score = val_metrics["net_sharpe"]
        if score > best_score:
            best_score = score
            best_bundle = {
                "model_name": model_name,
                "model": model,
                "calibrator": calibrator,
                "threshold": thr,
                "prob_q20": float(np.quantile(p_val, 0.2)),
                "prob_q40": float(np.quantile(p_val, 0.4)),
                "prob_q60": float(np.quantile(p_val, 0.6)),
                "prob_q80": float(np.quantile(p_val, 0.8)),
                "target_meta": target_meta,
                "metrics": {
                    "validation": val_metrics,
                    "test": test_metrics,
                },
            }

    if best_bundle is None:
        raise RuntimeError("No valid model bundle was produced for this split.")

    return {
        "feature_columns": X_cols,
        "target_meta": target_meta,
        "models": split_models,
        "best_bundle": best_bundle,
    }


def _train_walk_forward_models_from_df(
    df: pd.DataFrame,
    *,
    artifact_dir: Path,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
    target_mode: str = "binary",
    target_quantile: float = 0.70,
    side_filter: str = "all",
) -> dict[str, Any]:
    """Train models on walk-forward splits and persist best artifacts/report."""
    df = _coerce_side_filter(df, side_filter)
    if df.empty:
        raise ValueError(f"No rows available after applying side_filter={side_filter!r}")
    X_cols = _feature_columns(df)

    split_specs = make_split_specs(
        df,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
    )

    report: dict[str, Any] = {
        "dataset_rows": int(len(df)),
        "side_filter": side_filter,
        "target_mode": target_mode,
        "target_quantile": float(target_quantile) if target_mode == "large_winner" else float("nan"),
        "feature_columns": X_cols,
        "splits": [],
        "best": {},
    }

    best_bundle: dict[str, Any] | None = None
    best_score = -1e18

    for split in split_specs:
        tr_mask = split["train_mask"]
        va_mask = split["val_mask"]
        te_mask = split["test_mask"]

        train_df = df.loc[tr_mask].copy()
        val_df = df.loc[va_mask].copy()
        test_df = df.loc[te_mask].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            continue
        try:
            bundle_info = fit_best_model_bundle(
                train_df,
                val_df,
                test_df,
                target_mode=target_mode,
                target_quantile=target_quantile,
            )
        except (ValueError, RuntimeError):
            continue

        split_entry = {
            "split_id": split["split_id"],
            "split_type": split["split_type"],
            "train_range": split["train_label"],
            "val_range": split["val_label"],
            "test_range": split["test_label"],
            "target_meta": bundle_info["target_meta"],
            "models": bundle_info["models"],
        }

        report["splits"].append(split_entry)

        candidate_best = dict(bundle_info["best_bundle"])
        score = float(candidate_best["metrics"]["validation"]["net_sharpe"])
        if score > best_score:
            best_score = score
            best_bundle = {
                **candidate_best,
                "split_id": split["split_id"],
            }

    if best_bundle is None:
        raise RuntimeError("No valid walk-forward split/model was produced.")

    model_dir = artifact_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.joblib"
    cal_path = model_dir / "calibration.joblib"
    thr_path = model_dir / "selected_threshold.json"
    report_path = model_dir / "walk_forward_report.json"

    joblib.dump(best_bundle["model"], model_path)
    joblib.dump(best_bundle["calibrator"], cal_path)
    with thr_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(best_bundle["threshold"]),
                "prob_q20": float(best_bundle["prob_q20"]),
                "prob_q40": float(best_bundle["prob_q40"]),
                "prob_q60": float(best_bundle["prob_q60"]),
                "prob_q80": float(best_bundle["prob_q80"]),
                "model_name": best_bundle["model_name"],
                "split_id": int(best_bundle["split_id"]),
                "target_mode": str(best_bundle["target_meta"]["target_mode"]),
                "target_quantile": _to_serializable(best_bundle["target_meta"]["target_quantile"]),
                "train_large_winner_cut": _to_serializable(best_bundle["target_meta"]["train_large_winner_cut"]),
                "side_filter": side_filter,
            },
            f,
            indent=2,
        )

    report["best"] = {
        "model_name": best_bundle["model_name"],
        "split_id": int(best_bundle["split_id"]),
        "threshold": float(best_bundle["threshold"]),
        "prob_q20": float(best_bundle["prob_q20"]),
        "prob_q40": float(best_bundle["prob_q40"]),
        "prob_q60": float(best_bundle["prob_q60"]),
        "prob_q80": float(best_bundle["prob_q80"]),
        "target_meta": best_bundle["target_meta"],
        "metrics": best_bundle["metrics"],
        "artifacts": {
            "model_path": str(model_path),
            "calibration_path": str(cal_path),
            "threshold_path": str(thr_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, default=_to_serializable, indent=2)

    logger.info("Saved best model to %s", model_path)
    logger.info("Saved calibration object to %s", cal_path)
    logger.info("Saved threshold to %s", thr_path)
    logger.info("Saved report to %s", report_path)

    return {
        "best_model_path": model_path,
        "calibration_path": cal_path,
        "threshold_path": thr_path,
        "report_path": report_path,
        "report": report,
    }


def train_walk_forward_models(
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
    target_mode: str = "binary",
    target_quantile: float = 0.70,
    side_filter: str = "all",
    artifact_subdir: str = "models",
) -> dict[str, Any]:
    """Train models on the persisted dataset and save artifacts under ``artifact_subdir``."""
    df = _load_ml_dataset()
    config = load_config()
    artifact_dir = config.data_dir / artifact_subdir
    return _train_walk_forward_models_from_df(
        df,
        artifact_dir=artifact_dir,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
        target_mode=target_mode,
        target_quantile=target_quantile,
        side_filter=side_filter,
    )


def train_walk_forward_models_from_dataframe(
    df: pd.DataFrame,
    artifact_dir: str | Path,
    *,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
    target_mode: str = "binary",
    target_quantile: float = 0.70,
    side_filter: str = "all",
) -> dict[str, Any]:
    """Notebook-friendly entry point for walk-forward training on an in-memory dataset."""
    return _train_walk_forward_models_from_df(
        df.copy(),
        artifact_dir=Path(artifact_dir),
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
        target_mode=target_mode,
        target_quantile=target_quantile,
        side_filter=side_filter,
    )
