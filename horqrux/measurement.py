from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .gates import Z
from .ops import apply_gate
from .types import State, TargetIdx


def qubit_magnetization(state: State) -> Array:
    """Calculates the magnetization of each qubit.

    Args:
        state (Array): State upon which to operate.
    Returns:
        Array: Array of shape [N, ] with the magnetization / qubit
    """

    def qubit_magnetization(idx: TargetIdx) -> ArrayLike:
        projection = apply_gate(state, Z(idx))
        return jnp.real(jnp.dot(jnp.conj(state.flatten()), projection.flatten()))

    temp = [qubit_magnetization(idx) for idx in np.arange(state.ndim)]
    return jnp.stack(temp)


def total_magnetization(state: State, n_out: int = 1) -> Array:
    """Calculates the total magnetization as $\sum_i Z(qubit_i)$ of the given state

    Args:
        state (Array): State upon which to operate.
    Returns:
        Array: Total magnetization of the state.
    """
    magnetization = qubit_magnetization(state)
    return jnp.sum(magnetization.reshape((n_out, -1)), axis=-1)
