from __future__ import annotations

from typing import Tuple

import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from horqrux.types import State, TargetIdx

from .gates import NOT, Rx, Rz
from .ops import apply_gate


def global_entangling(state: State) -> State:
    """Globally entangles qubits by two consequtive shifted cnot layers, i.e.
    first layer has cnot (0, 1) and the second (1, 2), etc...

    Args:
        state (Array): State to be entangled.

    Returns:
        Array: Entangled state.
    """
    n_qubits = state.ndim

    # Getting indices of first CNOT layer
    even_idx = np.arange(0, n_qubits - 1, step=2, dtype=int)
    even_target = tuple(even_idx + 1)
    even_control = tuple(even_idx)

    # Getting indices of first CNOT layer
    uneven_idx = np.arange(1, n_qubits - 1, step=2, dtype=int)
    uneven_target = tuple(uneven_idx + 1)
    uneven_control = tuple(uneven_idx)

    # Applying CNOT layers
    state = apply_gate(
        state,
        [NOT(t_idx, c_idx) for t_idx, c_idx in zip(even_target, even_control)],
    )
    if len(uneven_idx) != 0:
        state = apply_gate(
            state,
            [NOT(t_idx, c_idx) for t_idx, c_idx in zip(uneven_target, uneven_control)],
        )
    return state


def variational(theta: ArrayLike, state: ArrayLike, target_idx: TargetIdx) -> Array:
    """A single hardware efficient variational layer comprised of Rz-Rx-Rz layers followed by a global entangling layer.

    Args:
        theta (Array): Angles of gates - should be an array of shape [3 x n_qubits].
        state (Array): State upon which the variational layer acts.
        target_idx (TargetIdx): Indices on which to apply the variational layer.

    Returns:
        Array: State after variational layer.
    """
    # First Rz layer

    state = apply_gate(state, [Rz(angle, idx) for angle, idx in zip(theta[0], target_idx)])  # type: ignore[arg-type]

    # Second Rx layer
    state = apply_gate(state, [Rx(angle, idx) for angle, idx in zip(theta[1], target_idx)])  # type: ignore[arg-type]

    # Third Rz layer
    state = apply_gate(state, [Rz(angle, idx) for angle, idx in zip(theta[2], target_idx)])  # type: ignore[arg-type]

    return global_entangling(state)


def n_variational(theta: Array, state: Array, target_idx: TargetIdx) -> Array:
    """A series of hardware efficient variational layers each comprised of Rz-Rx-Rz layers
      followed by a global entangling layer.
      Number of layers is inferred from shape of theta. Checkpoints each layer for memory reasons.

    Args:
        theta (Array): Angles of gates - should be an array of shape [n_layers, 3 x n_qubits].
        state (Array): State upon which the variational layer acts.
        target_idx (TargetIdx): Indices on which to apply the variational layer.

    Returns:
        Array: State after variational layers.
    """

    # We checkpoint for memory efficiency
    @jax.checkpoint
    def update_fn(carry: ArrayLike, theta_layer: ArrayLike) -> Tuple[ArrayLike, None]:
        return variational(theta_layer, carry, target_idx), None

    return jax.lax.scan(update_fn, state, theta)[0]
