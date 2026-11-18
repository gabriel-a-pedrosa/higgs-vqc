"""Predictions, classification metrics and the AMS."""

from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .circuit import expvals
from .data import BACKGROUND_LABEL, SIGNAL_LABEL


def predict_scores(circuit: Callable, X, theta) -> np.ndarray:
    """Circuit score <Z0> for every row of X."""
    return np.asarray(expvals(circuit, X, theta), dtype=float)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC AUC."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.where(scores >= 0.0, SIGNAL_LABEL, BACKGROUND_LABEL)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=SIGNAL_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=SIGNAL_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=SIGNAL_LABEL, zero_division=0),
    }
    # AUC needs both classes in y_true
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score((y_true == SIGNAL_LABEL).astype(int), scores)
    else:
        metrics["auc"] = float("nan")
    return metrics


B_REG_DEFAULT = 10.0  # background regularisation term from the challenge


def ams(s: float, b: float, b_reg: float = B_REG_DEFAULT) -> float:
    """Approximate Median Significance for selected signal and background weight."""
    if s <= 0.0:
        return 0.0
    # AMS = sqrt(2 * ((s + b + b_reg) * ln(1 + s / (b + b_reg)) - s))
    radicand = 2.0 * ((s + b + b_reg) * math.log1p(s / (b + b_reg)) - s)
    return math.sqrt(radicand) if radicand > 0.0 else 0.0


def _ams_vec(s: np.ndarray, b: np.ndarray, b_reg: float = B_REG_DEFAULT) -> np.ndarray:
    """Vectorised AMS."""
    s = np.asarray(s, float)
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        radicand = 2.0 * ((s + b + b_reg) * np.log1p(s / (b + b_reg)) - s)
    out = np.sqrt(np.clip(radicand, 0.0, None))
    out[s <= 0.0] = 0.0
    return out


def renormalize_weights(weights, y_true, s_total: float, b_total: float):
    """Rescale the weights of each class to the full-dataset totals."""
    # the hold-out is only a sample, so its weights are scaled up to sum to
    # s_total and b_total and the AMS estimates the whole population
    w = np.asarray(weights, float).copy()
    y = np.asarray(y_true, float)
    sig = y == SIGNAL_LABEL
    bkg = ~sig
    if w[sig].sum() > 0:
        w[sig] *= s_total / w[sig].sum()
    if w[bkg].sum() > 0:
        w[bkg] *= b_total / w[bkg].sum()
    return w


def max_ams(scores, y_true, weights, b_reg: float = B_REG_DEFAULT) -> Dict[str, float]:
    """Best AMS over all score thresholds."""
    scores = np.asarray(scores, float)
    y = np.asarray(y_true, float)
    w = np.asarray(weights, float)
    is_sig = (y == SIGNAL_LABEL).astype(float)

    # sort by score so each prefix is one possible cut
    order = np.argsort(-scores)
    w_ord = w[order]
    cum_s = np.cumsum(w_ord * is_sig[order])
    cum_b = np.cumsum(w_ord * (1.0 - is_sig[order]))
    ams_vals = _ams_vec(cum_s, cum_b, b_reg)

    k = int(np.argmax(ams_vals))
    total_s = cum_s[-1]
    return {
        "ams": float(ams_vals[k]),
        "threshold": float(scores[order][k]),
        "signal_efficiency": float(cum_s[k] / total_s) if total_s > 0 else 0.0,
        "selected_fraction": float((k + 1) / len(scores)),
    }
