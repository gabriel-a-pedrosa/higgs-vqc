"""Variational quantum classifier for the ATLAS Higgs Boson Challenge."""

from .config import VQCConfig, resolve_data_path
from .data import Dataset, load_dataset, weight_totals
from .circuit import build_circuit, expvals, init_params, param_shape
from .evaluate import (
    ams,
    compute_metrics,
    max_ams,
    predict_scores,
    renormalize_weights,
)
from .train import TrainResult, make_optimizer, train
