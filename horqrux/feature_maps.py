from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike

from .gates import Ry
from .ops import apply_gate
from .types import State, TargetIdx


def chebyshev(x: ArrayLike, state: State, target_idx: TargetIdx) -> State:
    """Chebyshev Tower feature map with encoding $f(x) = 2 \cdot i \cdot \cos (x)$.

    Args:
        x (float): Input coordinates of shape N_dims.
        state (Array): State to operate on.
        target_idx (TargetIdx): Tuple of Tuples describing which qubits to operate on.

    Returns:
        Array: Output state.
    """
    # To do N-D encoding, we calculate the encoding for every dimension for every idx
    # and then simply take the first len(target) / ndim elements along the first axis and stack
    # This does assume n_qubits is an integer multiple of dimensions.

    encoding = 2 * (jnp.asarray(target_idx) + 1) * jnp.arccos(x)
    n_qubits_per_dim = encoding.shape[0] // encoding.shape[1]
    # Column major flatten as we need dimensions after each other
    encoding = encoding[:n_qubits_per_dim].flatten("F")
    return apply_gate(state, [Ry(angle, idx) for angle, idx in zip(encoding, target_idx)])  # type: ignore[arg-type]


def product(
    x: ArrayLike,
    state: State,
    target_idx: TargetIdx,
    map_fn: Callable = jnp.arcsin,
) -> State:
    """Product feature map.

    Args:
        x (float): Input coordinate.
        state (Array): State to operate on.
        target_idx (TargetIdx): Tuple of Tuples describing which qubits to operate on.
        map_fn (Callable, optional): Mapping function for the coordinate. Defaults to jnp.arcsin.

    Returns:
        Array: Output state.
    """
    encoding = jnp.ones(len(target_idx)) * map_fn(x)
    return apply_gate(state, [Ry(encoding, idx) for idx in target_idx])  # type: ignore[arg-type]
