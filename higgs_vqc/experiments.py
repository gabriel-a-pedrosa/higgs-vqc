"""AMS experiments for the VQC, with a classical model as reference."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DEFAULT_FEATURES, VQCConfig
from .data import load_clean_frame
from .train import train


def best_config(n_features: int = 5, **overrides) -> VQCConfig:
    """VQC on the n most important features, with quantile preprocessing."""
    defaults = dict(
        features=list(DEFAULT_FEATURES[:n_features]),
        preprocessing="quantile",
        subsample=1000,
        num_layers=5,
        optimizer="rmsprop",
        stepsize=0.05,
        epochs=45,
    )
    defaults.update(overrides)
    return VQCConfig(**defaults)


def ensemble_ams(
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    holdout_size: int = 10_000,
    base: Optional[VQCConfig] = None,
    verbose: bool = False,
) -> Dict:
    """Train one VQC per seed, average their scores and maximise the AMS.

    A gradient boosting model is trained on each of the same subsamples, so
    the classical reference sees exactly the same data as the VQC.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    from .circuit import build_circuit
    from .data import BACKGROUND_LABEL, SIGNAL_LABEL, Dataset, _make_scaler, weight_totals
    from .evaluate import ams, max_ams, predict_scores, renormalize_weights

    base = base or best_config()
    frame = load_clean_frame(base)
    s_total, b_total = weight_totals(base)

    # fixed hold-out at the end, training subsamples come from the rest
    holdout = frame.tail(holdout_size)
    pool = frame.iloc[:-holdout_size]
    y_hold = holdout["Label"].map({"s": SIGNAL_LABEL, "b": BACKGROUND_LABEL}).to_numpy(float)
    w_hold = renormalize_weights(holdout["Weight"].to_numpy(float), y_hold, s_total, b_total)
    X_hold_raw = holdout[base.features].to_numpy(float)

    scores: List = []
    singles: List[float] = []
    running: List[float] = []
    classical: List[float] = []
    for seed in seeds:
        sub = pool.sample(base.subsample, random_state=seed)
        X_raw = sub[base.features].to_numpy(float)
        y = sub["Label"].map({"s": SIGNAL_LABEL, "b": BACKGROUND_LABEL}).to_numpy(float)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y, test_size=base.test_size, random_state=seed, stratify=y
        )
        scaler = _make_scaler(base.preprocessing, len(X_tr)).fit(X_tr)
        X_hold = scaler.transform(X_hold_raw)
        dataset = Dataset(
            scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te,
            scaler, list(base.features),
        )
        result = train(dataclasses.replace(base, seed=seed), dataset, verbose=False)
        s = predict_scores(build_circuit(base), X_hold, result.params)
        scores.append(s)
        singles.append(max_ams(s, y_hold, w_hold)["ams"])
        running.append(max_ams(np.mean(scores, axis=0), y_hold, w_hold)["ams"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gb = GradientBoostingClassifier(random_state=seed).fit(dataset.X_train, y_tr)
        sig = list(gb.classes_).index(SIGNAL_LABEL)
        classical.append(max_ams(2.0 * gb.predict_proba(X_hold)[:, sig] - 1.0, y_hold, w_hold)["ams"])

        if verbose:
            print(
                f"[{len(base.features)} features, seed {seed}] VQC {singles[-1]:.3f} | "
                f"ensemble({len(scores)}) {running[-1]:.3f} | gboost {classical[-1]:.3f}",
                flush=True,
            )

    return {
        "features": list(base.features),
        "ensemble_ams": running[-1] if running else 0.0,
        "single_ams": singles,
        "single_mean": float(np.mean(singles)) if singles else 0.0,
        "single_std": float(np.std(singles)) if singles else 0.0,
        "running_ensemble_ams": running,
        "gboost_ams": classical,
        "gboost_mean": float(np.mean(classical)) if classical else 0.0,
        "gboost_std": float(np.std(classical)) if classical else 0.0,
        "trivial_ams": ams(s_total, b_total),
        "n_models": len(list(seeds)),
        "config": base.to_dict(),
    }
