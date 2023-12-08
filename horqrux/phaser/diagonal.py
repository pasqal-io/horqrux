from __future__ import annotations

from functools import reduce
from itertools import chain

import jax.numpy as jnp
from chex import Array

from .utils import diagonal_kronecker, kron_AI, kron_IA


def diagonal_onebody_hamiltonian(Hi: Array, weights: Array, idx: list[int]) -> Array:
    # Generates diagonal of diagonal onebody hamiltonian terms.
    # Not pretty but it works...
    def diagonal_Hi(diagonal: Array, idx: int) -> Array:
        return kron_IA(kron_AI(diagonal, 2 ** (n_qubits - idx - 1)), 2**idx)

    n_qubits = max(idx) + 1  # +1 cause of index
    Hi_diag = jnp.diag(Hi)
    return reduce(
        lambda state, x: state + x[0] * diagonal_Hi(Hi_diag, x[1]),
        zip(weights, idx),
        jnp.zeros(2**n_qubits),
    )


def diagonal_twobody_hamiltonian(
    HiHj: tuple[Array, Array], weights: Array, idx: list[tuple[int, int]]
) -> Array:
    # Generates diagonal of diagonal two-body hamiltonian terms.
    # Not pretty but it works...
    def diagonal_Hi(diagonal: list[Array], idx_ij: tuple[int, int]) -> Array:
        idx_i, idx_j = idx_ij
        left = kron_IA(diagonal[0], 2 ** (idx_i))
        right = kron_IA(kron_AI(diagonal[1], 2 ** (n_qubits - idx_j - 1)), 2 ** (idx_j - idx_i - 1))
        return diagonal_kronecker(left, right)

    n_qubits = max(list(chain(*idx))) + 1  # +1 cause of index
    HiHj_diag = [jnp.diag(H) for H in HiHj]
    return reduce(
        lambda state, x: state + x[0] * diagonal_Hi(HiHj_diag, x[1]),
        zip(weights, idx),
        jnp.zeros(2**n_qubits),
    )
