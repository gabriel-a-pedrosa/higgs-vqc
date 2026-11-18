"""Configuration for a VQC run."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# the five most important features from the ranking, most important first
DEFAULT_FEATURES: List[str] = [
    "DER_mass_MMC",
    "DER_mass_vis",
    "DER_mass_transverse_met_lep",
    "PRI_tau_pt",
    "DER_deltar_tau_lep",
]

# ATLAS uses -999.0 for quantities that could not be computed.
# Rows with this value in a selected feature are dropped.
MAGIC_MISSING_VALUE = -999.0

_DATA_FILENAME = "atlas-higgs-challenge-2014-v2.csv"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_path(explicit: Optional[str | Path] = None) -> Path:
    """Find the ATLAS Higgs CSV or raise FileNotFoundError."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("HIGGS_CSV")
    if env:
        candidates.append(Path(env))
    candidates.append(_REPO_ROOT / "data" / _DATA_FILENAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find '{_DATA_FILENAME}'. Looked in:\n  - {searched}\n"
        "Download it (see data/README.md) or set the HIGGS_CSV environment "
        "variable to its path."
    )


@dataclass(frozen=True)
class VQCConfig:
    """Hyper-parameters for one VQC run."""

    # data
    features: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    subsample: Optional[int] = 100
    test_size: float = 0.30
    seed: int = 42
    preprocessing: str = "minmax"  # "minmax" or "quantile"

    # circuit
    num_layers: int = 5
    fixed_angles: List[float] = field(
        default_factory=lambda: [0.0, 0.0, math.pi / 2]
    )
    ring: bool = True  # extra CNOT from the last qubit back to qubit 0

    # optimisation
    optimizer: str = "rmsprop"  # adam, rmsprop or gd
    stepsize: float = 0.05
    epochs: int = 50
    batch_size: Optional[int] = 20  # None means full batch

    eval_every: int = 5
    output_dir: Path = field(default_factory=lambda: _REPO_ROOT / "experiments")

    @property
    def n_qubits(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["n_qubits"] = self.n_qubits
        return d
