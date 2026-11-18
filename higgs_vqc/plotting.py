"""Figures for the README."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")


def _save(fig, out_dir: str | Path, name: str, tight: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_ranking(result: dict, out_dir: str | Path, top: int = 15,
                         chosen: int = 5) -> Path:
    """Permutation importance of the most important features."""
    rows = result["ranking"][:top]
    names = [r["feature"] for r in rows][::-1]
    vals = [r["auc_drop"] for r in rows][::-1]
    errs = [r["std"] for r in rows][::-1]
    colors = ["C0" if i >= len(rows) - chosen else "lightgrey" for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(8.0, 0.32 * len(rows) + 1.2))
    ax.barh(names, vals, xerr=errs, color=colors)
    ax.set_xlabel("Drop in test ROC AUC when the feature is shuffled")
    ax.set_title("Feature importance, gradient boosting on all 30 features")
    ax.tick_params(axis="y", labelsize=9)
    return _save(fig, out_dir, "feature_ranking.png")


def plot_ams_results(steps, out_dir: str | Path, trivial: float,
                     classical_ref: Optional[float] = None) -> Path:
    """Bar chart of the VQC AMS for each setup, steps given as (label, ams, err)."""
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    errs = [s[2] for s in steps]
    y = np.arange(len(steps))[::-1]  # first step on top
    colors = sns.color_palette("crest", len(steps))

    fig, ax = plt.subplots(figsize=(8.0, 0.7 * len(steps) + 1.5))
    ax.barh(y, values, xerr=errs, color=colors)
    for yi, v, e in zip(y, values, errs):
        ax.text(v + (e or 0) + 0.03, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(trivial, ls=":", color="grey", label=f"select all ({trivial:.2f} σ)")
    if classical_ref is not None:
        ax.axvline(classical_ref, ls="--", color="C3",
                   label=f"gradient boosting ({classical_ref:.2f} σ)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Max AMS (σ)")
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title("VQC maximum AMS on a 10,000-event hold-out")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    return _save(fig, out_dir, "ams_results.png")


def plot_circuit(config, out_dir: str | Path) -> Path:
    """Draw the encoding and one layer, since every layer has the same gates."""
    import dataclasses

    import pennylane as qml
    from .circuit import build_circuit, init_params

    one_layer = dataclasses.replace(config, num_layers=1)
    circuit = build_circuit(one_layer)
    params = init_params(one_layer)
    x = np.full(config.n_qubits, np.pi / 2)
    fig, _ = qml.draw_mpl(circuit)(x, params)
    fig.suptitle(f"Encoding and one of the {config.num_layers} layers ({config.n_qubits} qubits)")
    return _save(fig, out_dir, "circuit.png", tight=False)
