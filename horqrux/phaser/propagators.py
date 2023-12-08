from __future__ import annotations

from functools import reduce

import jax.numpy as jnp
import numpy as np
from chex import Array

from .utils import hilbert_reshape


def first_order_trotter(H: list, state, dt, weights=None):
    # First order trotter
    if weights is None:
        weights = len(H) * [None]

    return reduce(
        lambda state, x: x[0].evolve(state, dt, x[1]),
        zip(H, weights),
        state,
    )


def second_order_trotter(H: list, state, dt, weights=None):
    # second order trotter
    if weights is None:
        weights = len(H) * [None]

    return reduce(
        lambda state, x: x[0].evolve(state, dt / 2, x[1]),
        zip([*H[::-1], *H], [*weights[::-1], *weights]),
        state,
    )


def apply_unitary(state: Array, U: Array, target_idx: tuple) -> Array:
    def _apply_diagonal_unitary(state: Array, U: Array, target_idx: tuple) -> Array:
        return U.reshape(state.shape) * state

    def _apply_matrix_unitary(state: Array, U: Array, target_idx: tuple) -> Array:
        if len(target_idx) > 1:
            U = hilbert_reshape(U)

        # Move axis to front, operate, move back
        state = jnp.moveaxis(state, target_idx, np.arange(len(target_idx)))
        state = jnp.tensordot(U, state, axes=len(target_idx))
        return jnp.moveaxis(state, np.arange(len(target_idx)), target_idx)

    if U.ndim == 1:
        return _apply_diagonal_unitary(state, U, target_idx)
    elif U.ndim == 2:
        return _apply_matrix_unitary(state, U, target_idx)
    else:
        raise NotImplementedError
