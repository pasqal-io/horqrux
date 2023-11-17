# This shows how to build an efficient model using diagonalization
from __future__ import annotations

from time import time

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from chex import Array
from jax import random
from phaser.diagonal import diagonal_onebody_hamiltonian, diagonal_twobody_hamiltonian
from phaser.hamiltonians import HamiltonianTerm, Pauli_x, n
from phaser.propagators import second_order_trotter
from phaser.simulate import simulate
from phaser.utils import init_state, kron_sum

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


class DiagonalRydbergHamiltonian(nn.Module):
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


if __name__ == "__main__":
    # Initializing Hamiltonian
    n_qubits = 20
    dt, N = 1e-3, 3000
    laser_params = (1.0, 2.0)
    U = jnp.triu(random.normal(key, (n_qubits, n_qubits)) ** 2)
    in_state = init_state(n_qubits)

    def laser(laser_params, t):
        (w_rabi, w_detune) = laser_params
        return {
            "rabi": 20.0 * jnp.cos(2 * jnp.pi * w_rabi * t),
            "detune": 15.0 * jnp.cos(2 * jnp.pi * w_detune * t),
        }

    hamiltonian = DiagonalRydbergHamiltonian(n_qubits, U)
    hamiltonian_params = hamiltonian.init(
        key,
        in_state,
        laser(laser_params, 0),
    )

    # Timing
    start = time()
    _ = simulate(
        hamiltonian,
        hamiltonian_params,
        laser,
        laser_params,
        N,
        dt,
        in_state,
    ).block_until_ready()
    stop = time()

    print(f"Simulation time for {n_qubits} qubits and {N} steps: {stop - start}s")
    print(
        "Note that for clarity we didn't jit the final function, so compilation time is included."
    )
