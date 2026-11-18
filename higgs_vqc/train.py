"""Training loop for the VQC."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .circuit import build_circuit, expvals, init_params
from .config import VQCConfig
from .data import Dataset, load_dataset
from .evaluate import compute_metrics, predict_scores

_OPTIMIZERS = {
    "adam": qml.AdamOptimizer,
    "rmsprop": qml.RMSPropOptimizer,
    "gd": qml.GradientDescentOptimizer,
}


def make_optimizer(config: VQCConfig):
    """PennyLane optimizer named in the config."""
    try:
        cls = _OPTIMIZERS[config.optimizer.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown optimizer '{config.optimizer}'. "
            f"Choose from: {', '.join(_OPTIMIZERS)}"
        )
    return cls(stepsize=config.stepsize)


def _iter_batches(X, y, batch_size: Optional[int], rng: np.random.Generator):
    """Shuffled mini-batches, batch_size None means full batch."""
    n = len(X)
    idx = rng.permutation(n)
    if batch_size is None:
        batch_size = n
    for start in range(0, n, batch_size):
        chunk = idx[start : start + batch_size]
        yield X[chunk], y[chunk]


@dataclass
class TrainResult:
    """Weights, history and timing of one training run."""

    config: VQCConfig
    params: np.ndarray
    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_test_accuracy: float = -1.0
    best_params: Optional[np.ndarray] = None
    train_seconds: float = 0.0

    @property
    def final(self) -> Dict[str, float]:
        return self.history[-1] if self.history else {}


def train(
    config: VQCConfig,
    dataset: Optional[Dataset] = None,
    verbose: bool = True,
) -> TrainResult:
    """Train a VQC and return the result with the per-epoch history."""
    if dataset is None:
        dataset = load_dataset(config)

    circuit = build_circuit(config)
    params = init_params(config)
    opt = make_optimizer(config)
    rng = np.random.default_rng(config.seed)

    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    def cost(theta, Xb, yb):
        preds = expvals(circuit, Xb, theta)
        return pnp.mean((preds - yb) ** 2)

    result = TrainResult(config=config, params=params)
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        for Xb, yb in _iter_batches(X_train, y_train, config.batch_size, rng):
            Xb_q = pnp.array(Xb)
            yb_q = pnp.array(yb)
            params = opt.step(lambda t: cost(t, Xb_q, yb_q), params)

        train_scores = predict_scores(circuit, X_train, params)
        test_scores = predict_scores(circuit, X_test, params)
        train_m = compute_metrics(y_train, train_scores)
        test_m = compute_metrics(y_test, test_scores)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean((train_scores - y_train) ** 2)),
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }
        result.history.append(record)

        if test_m["accuracy"] > result.best_test_accuracy:
            result.best_test_accuracy = test_m["accuracy"]
            result.best_epoch = epoch
            result.best_params = np.array(params)

        if verbose and (epoch % config.eval_every == 0 or epoch == config.epochs):
            print(
                f"Epoch {epoch:03d} | loss {record['train_loss']:.3f} "
                f"| train acc {train_m['accuracy']:.3f} "
                f"| test acc {test_m['accuracy']:.3f} "
                f"| test F1 {test_m['f1']:.3f} | test AUC {test_m['auc']:.3f}"
            )

    result.params = np.array(params)
    result.train_seconds = time.time() - start
    return result
