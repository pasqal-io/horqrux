from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random

from horqrux.phaser.diagonal import diagonal_onebody_hamiltonian, diagonal_twobody_hamiltonian
from horqrux.phaser.hamiltonians import Interaction, Number, n
from horqrux.phaser.utils import diagonal_kronecker, make_explicit

key = random.PRNGKey(42)


def test_diagonal_kronecker():
    A = jnp.array([1, 3])
    B = jnp.array([2, 1])

    assert jnp.allclose(diagonal_kronecker(A, B), jnp.diag(jnp.kron(jnp.diag(A), jnp.diag(B))))


def test_onebody_hamiltonian():
    n_qubits = 6
    detune_weights = random.normal(key, (n_qubits,))
    H_detune = list(map(lambda x: Number((x,), None), range(n_qubits)))
    H_detune = make_explicit(H_detune, detune_weights, n_qubits)

    H_detune_diagonal = diagonal_onebody_hamiltonian(n, detune_weights, np.arange(n_qubits))
    assert jnp.allclose(H_detune_diagonal, jnp.diag(H_detune))


def test_twobody_hamiltonian():
    # This test currently fails as H_fn doesnt give back two H's.
    n_qubits = 6

    U = random.normal(key, (n_qubits, n_qubits))
    U = jnp.triu(U**2, k=1)
    idx = list(zip(*np.triu_indices_from(U, k=1)))
    H_interaction = list(map(lambda x: Interaction(x, None), idx))

    weights = U[jnp.triu_indices_from(U, k=1)]
    H_interaction = make_explicit(H_interaction, weights, n_qubits)

    H_interaction_diagonal = diagonal_twobody_hamiltonian([n, n], weights, idx)

    assert jnp.allclose(H_interaction_diagonal, jnp.diag(H_interaction))


if __name__ == "__main__":
    test_onebody_hamiltonian()
