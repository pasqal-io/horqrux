from __future__ import annotations

from typing import Iterable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def prepare_state(n_qubits: int, state: str = None) -> Array:
    """Generates a state of shape [2, 2, ..]_n_qubits in the given state.
    If given state is None, return all qubits in |0> state.

    Args:
        n_qubits (int): size of state.
        state (str, optional): Bitstring describing state to be prepared. Defaults to None.

    Returns:
        Array: Prepared state.
    """
    if state is None:
        state = "0" * n_qubits

    space = jnp.zeros(tuple(2 for _ in range(n_qubits)), dtype=jnp.complex128)
    space = space.at[tuple(map(int, state))].set(1.0)
    return space


def none_like(x: Iterable) -> Tuple[None, ...]:
    """Generates a tuple of Nones with equal length to x. Useful for gates with multiple targets but no control.

    Args:
        x (Iterable): Iterable to be mimicked.

    Returns:
        Tuple[None, ...]: Tuple of Nones of length x.
    """
    return tuple(map(lambda _: None, x))


def hilbert_reshape(O: ArrayLike) -> Array:
    """Reshapes O of shape [M, M] to array of shape [2, 2, ...].
       Useful for working with controlled and multi-qubit gates.

    Args:
        O (Array): Array to be reshaped.

    Returns:
        Array: Array with shape [2, 2, ...]
    """
    n_axes = int(np.log2(O.size))
    return O.reshape(tuple(2 for _ in np.arange(n_axes)))


def equivalent_state(state: Array, reference_state: str) -> bool:
    """Small utility to easily compare an output state to a given (pure) state like
        equivalent_state(output_state, '10').

    Args:
        state (Array): State array to be checked.
        reference_state (str): Bitstring to compare the state to.

    Returns:
        bool: Boolean indicating whether the states are equivalent.
    """
    n_qubits = state.ndim
    ref_state = prepare_state(n_qubits, reference_state)
    return jnp.allclose(state, ref_state)  # type: ignore[no-any-return]


def overlap(state: Array, projection: Array) -> Array:
    return jnp.real(jnp.dot(jnp.conj(state.flatten()), projection.flatten()))
