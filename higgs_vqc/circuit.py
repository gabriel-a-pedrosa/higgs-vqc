"""The variational circuit."""

from __future__ import annotations

from typing import Callable, Tuple

import pennylane as qml
from pennylane import numpy as pnp

from .config import VQCConfig

ROTATIONS_PER_QUBIT = 3


def param_shape(config: VQCConfig) -> Tuple[int, int, int]:
    """Shape of the trainable weights (layers, qubits, 3)."""
    return (config.num_layers, config.n_qubits, ROTATIONS_PER_QUBIT)


def init_params(config: VQCConfig) -> pnp.ndarray:
    """Small random initial weights."""
    rng = pnp.random.default_rng(config.seed)
    weights = 0.01 * rng.standard_normal(param_shape(config))
    return pnp.array(weights, requires_grad=True)


def build_circuit(config: VQCConfig) -> Callable:
    """Return the data-reuploading VQC as a QNode."""
    n_qubits = config.n_qubits
    fixed = config.fixed_angles
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, theta):
        # x[..., i] works for a single sample and for a broadcast batch
        for i in range(n_qubits):
            qml.RY(x[..., i], wires=i)

        # fixed, non-trainable layer
        for i in range(n_qubits):
            qml.RY(fixed[0], wires=i)
            qml.RX(fixed[1], wires=i)
            qml.RZ(fixed[2], wires=i)

        for layer in range(config.num_layers):
            for i in range(n_qubits):
                qml.RY(x[..., i], wires=i)  # reupload the features
                qml.RX(theta[layer, i, 0], wires=i)
                qml.RY(theta[layer, i, 1], wires=i)
                qml.RZ(theta[layer, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if config.ring and n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        return qml.expval(qml.PauliZ(0))

    return circuit


def expvals(circuit: Callable, X, theta) -> pnp.ndarray:
    """Evaluate the circuit on a batch and return a 1D array of <Z0>."""
    X = pnp.atleast_2d(X)
    out = circuit(X, theta)
    return pnp.reshape(out, (-1,))
