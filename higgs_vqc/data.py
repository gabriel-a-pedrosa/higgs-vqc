"""Loading and preprocessing of the ATLAS Higgs dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from .config import MAGIC_MISSING_VALUE, VQCConfig, resolve_data_path


def _make_scaler(preprocessing: str, n_train: int):
    """Scaler that maps the features to [0, pi] for angle encoding."""
    to_angles = MinMaxScaler(feature_range=(0.0, np.pi))
    if preprocessing == "minmax":
        return to_angles
    if preprocessing == "quantile":
        # rank transform first so the long-tailed masses are not squashed by outliers
        n_q = int(max(10, min(1000, n_train)))
        quantile = QuantileTransformer(output_distribution="uniform", n_quantiles=n_q)
        return Pipeline([("quantile", quantile), ("angles", to_angles)])
    raise ValueError(f"Unknown preprocessing '{preprocessing}'.")

# labels match the eigenvalues of Z so sign(<Z>) is the predicted class
SIGNAL_LABEL = 1.0
BACKGROUND_LABEL = -1.0


@dataclass
class Dataset:
    """Train/test split and the scaler used to make it."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    feature_names: list[str]

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


# avoid re-reading the 195 MB CSV for every run
_FRAME_CACHE: dict[tuple, pd.DataFrame] = {}


def load_clean_frame(config: VQCConfig, data_path: Optional[Path] = None) -> pd.DataFrame:
    """Read the CSV once, drop rows with missing values and cache it."""
    path = resolve_data_path(data_path)
    key = (str(path), tuple(config.features))
    if key not in _FRAME_CACHE:
        available = pd.read_csv(path, nrows=0).columns
        cols = config.features + ["Label"]
        if "Weight" in available:
            cols = cols + ["Weight"]
        df = pd.read_csv(path, usecols=cols)
        df = df.replace(MAGIC_MISSING_VALUE, np.nan).dropna(subset=config.features)
        _FRAME_CACHE[key] = df.reset_index(drop=True)
    return _FRAME_CACHE[key]


def weight_totals(config: VQCConfig, data_path: Optional[Path] = None) -> tuple[float, float]:
    """Total signal and background weight over the cleaned dataset."""
    df = load_clean_frame(config, data_path)
    if "Weight" not in df.columns:
        raise KeyError("The dataset has no 'Weight' column; AMS cannot be computed.")
    s_total = float(df.loc[df["Label"] == "s", "Weight"].sum())
    b_total = float(df.loc[df["Label"] == "b", "Weight"].sum())
    return s_total, b_total


def _read_clean_frame(config: VQCConfig, data_path: Optional[Path]) -> pd.DataFrame:
    """Cleaned frame, subsampled with the config seed."""
    df = load_clean_frame(config, data_path)
    if config.subsample is not None and config.subsample < len(df):
        df = df.sample(config.subsample, random_state=config.seed)
    return df


def load_dataset(
    config: VQCConfig,
    data_path: Optional[Path] = None,
    frame: Optional[pd.DataFrame] = None,
) -> Dataset:
    """Load, clean, scale to [0, pi] and split the dataset."""
    if frame is not None:
        df = frame
        if config.subsample is not None and config.subsample < len(df):
            df = df.sample(config.subsample, random_state=config.seed)
    else:
        df = _read_clean_frame(config, data_path)

    y = df["Label"].map({"s": SIGNAL_LABEL, "b": BACKGROUND_LABEL}).to_numpy(float)
    X_raw = df[config.features].to_numpy(float)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=config.test_size, random_state=config.seed, stratify=y
    )

    # fit the scaler on the training split only
    scaler = _make_scaler(config.preprocessing, len(X_train_raw)).fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=list(config.features),
    )
