"""Rank the 30 ATLAS features before choosing which ones go into the VQC."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import MAGIC_MISSING_VALUE, resolve_data_path


def feature_ranking(
    n_events: int = 200_000,
    n_repeats: int = 5,
    seed: int = 0,
    exclude_last: int = 20_000,
    data_path: Optional[str | Path] = None,
) -> dict:
    """Permutation importance of every feature for a gradient boosting model.

    The importance of a feature is how much the test ROC AUC drops when that
    feature is shuffled. Missing values (-999) are left as NaN, which the
    model handles directly.
    """
    df = pd.read_csv(resolve_data_path(data_path))
    df = df.iloc[:-exclude_last]  # the VQC hold-out comes from the end of the file
    features = [c for c in df.columns if c.startswith(("DER_", "PRI_"))]
    X = df[features].replace(MAGIC_MISSING_VALUE, np.nan)
    y = (df["Label"] == "s").astype(int)

    X, _, y, _ = train_test_split(X, y, train_size=n_events, random_state=seed, stratify=y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    perm = permutation_importance(
        model, X_te, y_te, scoring="roc_auc", n_repeats=n_repeats, random_state=seed
    )
    order = np.argsort(perm.importances_mean)[::-1]
    ranking = [
        {
            "feature": features[i],
            "auc_drop": float(perm.importances_mean[i]),
            "std": float(perm.importances_std[i]),
        }
        for i in order
    ]
    return {"test_auc_all_features": float(auc), "n_events": n_events, "ranking": ranking}


def top_features(result: dict, k: int) -> list[str]:
    """The k most important features, most important first."""
    return [row["feature"] for row in result["ranking"][:k]]
