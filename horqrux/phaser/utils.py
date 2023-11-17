from __future__ import annotations

from functools import reduce
from typing import Any

import jax.numpy as jnp
import numpy as np
from chex import Array


def hilbert_reshape(U: Array) -> Array:
    # Reshapes O of shape [M, M] to array of shape [2, 2, ...]. Useful for working with controlled and multi-qubit gates.
    n_axes = int(np.log2(U.size))
    return U.reshape(tuple(2 for _ in np.arange(n_axes)))


def kron_prod(*x: Array) -> Array:
    # custom kronecker product which can multiply multiple matrices
    # and filters out zeros because sometime we generate jnp.eye(0)
    return reduce(jnp.kron, filter(lambda x: x.size != 0, x))


def kron_sum(H: list, state: Array, weights: list = None) -> Array:
    if weights is None:
        weights = len(H) * [None]
    # kronecker sum
    return reduce(
        lambda out_state, x: out_state + x[0](state, x[1]),
        zip(H, weights),
        jnp.zeros_like(state),
    )


def make_explicit(H: Array, params: list, n_qubits: int) -> Array:
    def Hi(x: Any) -> Any:
        term, param = x
        idx = term.idx
        _H = term.H(idx, param)

        # 1 body term
        if len(idx) == 1:
            (idx,) = idx
            return kron_prod(jnp.eye(2**idx), _H, jnp.eye(2 ** (n_qubits - idx - 1)))
        # 2 body term
        elif len(idx) == 2:
            idx_i, idx_j = idx
            return kron_prod(
                jnp.eye(2 ** (idx_i)),
                _H[0],
                jnp.eye(2 ** (idx_j - idx_i - 1)),
                _H[1],
                jnp.eye(2 ** (n_qubits - idx_j - 1)),
            )

    _H = jnp.zeros((2**n_qubits, 2**n_qubits))
    return reduce(lambda H_full, x: H_full + Hi(x), zip(H, params), _H)


def init_state(n_qubits: int) -> Array:
    state = jnp.zeros(tuple(2 for _ in np.arange(n_qubits)), dtype=jnp.complex64)
    state = state.at[tuple(-1 for _ in np.arange(n_qubits))].set(1.0)
    return state


def diagonal_kronecker(A: Array, B: Array) -> Array:
    """Given two diagonal A and B, calculates diagonal of kronecker product,
    which is also diagonal."""
    return (A[:, None] * B[None, :]).reshape(A.size * B.size)


def kron_AI(A: Array, N: int) -> Array:
    # Calculates A kron I, diagonal only.
    return jnp.repeat(A, repeats=N, axis=0)


def kron_IA(A: Array, N: int) -> Array:
    # Calculates I kron A, diagonal only.
    return jnp.tile(A, reps=N)
