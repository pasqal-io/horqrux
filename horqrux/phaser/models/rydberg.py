from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from chex import Array
from jax import random
from horqrux.phaser.diagonal import diagonal_onebody_hamiltonian, diagonal_twobody_hamiltonian
from horqrux.phaser.hamiltonians import HamiltonianTerm, Pauli_x, n
from horqrux.phaser.propagators import second_order_trotter
from horqrux.phaser.utils import kron_sum

key = random.PRNGKey(42)


# Defining diagonal detuning
def diagonal_detune_H(idx, weights):
    return diagonal_onebody_hamiltonian(n, weights, idx)


def diagonal_detune_expm(idx, weights):
    return jnp.exp(-1j * diagonal_detune_H(idx, weights))


DiagonalDetune = HamiltonianTerm.create(diagonal_detune_H, diagonal_detune_expm)


# Interaction
def diagonal_interaction_H(idx, weights):
    return diagonal_twobody_hamiltonian((n, n), weights, idx)


def diagonal_interaction_expm(idx, weights):
    return jnp.exp(-1j * diagonal_interaction_H(idx, weights))


DiagonalInteraction = HamiltonianTerm.create(diagonal_interaction_H, diagonal_interaction_expm)


def generate_interaction(U):
    U_params = jnp.stack(U[np.triu_indices_from(U, k=1)])
    idx = tuple(zip(*np.triu_indices_from(U, k=1)))

    return DiagonalInteraction(idx, lambda key: U_params)


class RydbergHamiltonian(nn.Module):
    n_qubits: int
    U: Array

    def setup(self):
        # Rabi terms
        H_rabi = [Pauli_x((idx,), None) for idx in range(self.n_qubits)]

        # Detuning
        H_detune = DiagonalDetune(range(self.n_qubits), None)

        # Interaction term
        H_interact = generate_interaction(self.U)

        # Joining all terms
        self.H = [*H_rabi, H_detune, H_interact]

    def __call__(self, state, weights):
        return kron_sum(self.H, state, self.parse_weights(weights))

    def evolve(self, state: Array, dt: float, weights: dict):
        return second_order_trotter(self.H, state, dt, self.parse_weights(weights))

    def parse_weights(self, weights):
        # Parse the weights from tuple to correct shape and values
        return [
            *jnp.full((self.n_qubits,), weights["rabi"] / 2),
            jnp.full((self.n_qubits,), -weights["detune"]),
            None,
        ]
