from __future__ import annotations

import numpy as np
from jax import Array
from jax.typing import ArrayLike

from horqrux.feature_maps import chebyshev
from horqrux.measurement import total_magnetization
from horqrux.variational_ansatz import n_variational

from .types import State


def DQC(state: State, theta: ArrayLike, x: ArrayLike, n_out: int) -> Array:
    """Basic DQC function with chebyshev feature encoding, n variational layers
    and total magnetization as measurement

    !!! warning
        Make sure eveything is 64 bits!

    Args:
        state (State): Input state vector of shape [2, 2, ... n_qubits]
        theta (Array): Angles of variational layers of shape [N_layers, 3, N_qubits]
        x (float): Input coordinate
        n_out (int): Number of outputs.

    Returns:
        Array: Total magnetization.
    """
    n_qubits = state.ndim
    q_idx = tuple((idx,) for idx in np.arange(n_qubits))

    state = chebyshev(x, state, q_idx)
    state = n_variational(theta, state, q_idx)
    return total_magnetization(state, n_out)
