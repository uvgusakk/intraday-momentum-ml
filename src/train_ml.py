"""Walk-forward training utilities for trade-filter classifiers."""

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


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"date", "timestamp", "side", "y", "pnl", "trade_return", "net_pnl", "return", "costs"}
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


def train_walk_forward_models(
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
) -> dict[str, Any]:
    """Train models on walk-forward splits and persist best artifacts/report."""
    df = _load_ml_dataset()
    X_cols = _feature_columns(df)

    returns_all = _trade_return_series(df)
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

    report: dict[str, Any] = {
        "dataset_rows": int(len(df)),
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
        if train_df["y"].nunique() < 2 or val_df["y"].nunique() < 2:
            continue

        X_train, y_train = train_df[X_cols], train_df["y"].astype(int)
        X_val, y_val = val_df[X_cols], val_df["y"].astype(int)
        X_test, y_test = test_df[X_cols], test_df["y"].astype(int)

        ret_val = returns_all.loc[val_df.index]
        ret_test = returns_all.loc[test_df.index]

        models = {
            "logistic": train_logistic_regression(X_train, y_train),
            "lightgbm": train_lightgbm_classifier(X_train, y_train, X_val=X_val, y_val=y_val),
        }

        split_entry: dict[str, Any] = {
            "split_id": split["split_id"],
            "split_type": split["split_type"],
            "train_range": split["train_label"],
            "val_range": split["val_label"],
            "test_range": split["test_label"],
            "models": {},
        }

        for model_name, model in models.items():
            val_scores = _raw_scores(model, X_val)
            calibrator = PlattCalibrator().fit(val_scores, y_val.to_numpy())

            p_val = calibrator.predict_proba(val_scores)
            thr, val_metrics = _select_threshold(y_val, p_val, ret_val)

            test_scores = _raw_scores(model, X_test)
            p_test = calibrator.predict_proba(test_scores)
            test_metrics = _classification_metrics(y_test, p_test, thr)
            test_metrics.update(_trading_metrics(ret_test, p_test, thr))

            split_entry["models"][model_name] = {
                "threshold": thr,
                "validation": val_metrics,
                "test": test_metrics,
            }

            # Production selection must use validation only (avoid test leakage).
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
                    "split_id": split["split_id"],
                    "metrics": {
                        "validation": val_metrics,
                        "test": test_metrics,
                    },
                }

        report["splits"].append(split_entry)

    if best_bundle is None:
        raise RuntimeError("No valid walk-forward split/model was produced.")

    config = load_config()
    model_dir = config.data_dir / "models"
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
