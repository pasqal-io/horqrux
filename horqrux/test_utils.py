from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .utils import prepare_state


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
