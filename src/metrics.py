"""Performance and classification metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def summarize_backtest(returns: pd.Series) -> dict[str, float]:
    """Compute core strategy metrics from return series.

    Args:
        returns: Daily strategy return series.

    Returns:
        dict[str, float]: Summary metrics including sharpe, cagr_ish and max_drawdown.
    """
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return {
            "sharpe": 0.0,
            "cagr_ish": 0.0,
            "max_drawdown": 0.0,
            "n_days": 0.0,
        }

    mu = float(clean.mean())
    sigma = float(clean.std(ddof=0))
    sharpe = 0.0 if sigma == 0.0 else (mu / sigma) * np.sqrt(252.0)

    equity = (1.0 + clean).cumprod()
    periods = len(clean)
    years = periods / 252.0
    cagr_ish = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "cagr_ish": cagr_ish,
        "max_drawdown": max_drawdown(equity),
        "n_days": float(periods),
    }


def summarize_classifier(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, Any]:
    """Compute classification performance metrics."""
    y_true_clean = pd.Series(y_true).astype(int)
    y_pred_clean = pd.Series(y_pred).astype(int)
    y_proba_clean = pd.Series(y_proba).astype(float)

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_clean, y_pred_clean)),
        "precision": float(precision_score(y_true_clean, y_pred_clean, zero_division=0)),
        "recall": float(recall_score(y_true_clean, y_pred_clean, zero_division=0)),
    }

    try:
        out["roc_auc"] = float(roc_auc_score(y_true_clean, y_proba_clean))
    except ValueError:
        out["roc_auc"] = float("nan")

    return out
