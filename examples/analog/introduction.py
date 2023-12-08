# This example shows how to build a model hamiltonian and simulate it.
from __future__ import annotations

from time import time

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from chex import Array
from jax import random
from phaser.hamiltonians import Interaction, Number, Pauli_x
from phaser.propagators import second_order_trotter
from phaser.simulate import simulate
from phaser.utils import init_state, kron_sum

key = random.PRNGKey(42)


class RydbergHamiltonian(nn.Module):
    n_qubits: int
    U: Array

    def setup(self):
        # Rabi terms
        H_rabi = [Pauli_x((idx,), None) for idx in np.arange(self.n_qubits)]

        # Detuning terms
        H_detune = [Number((idx,), None) for idx in np.arange(self.n_qubits)]

        # Interaction term
        # We don't want to learn U here so it's just a matrix
        self.U_params = self.U[np.triu_indices_from(self.U, k=1)]
        H_interact = [Interaction(idx, None) for idx in zip(*np.triu_indices_from(self.U, k=1))]

        # Joining all terms
        self.H = H_rabi + H_detune + H_interact

    def __call__(self, state, weights):
        weights = jnp.concatenate([weights["rabi"] / 2, -weights["detune"], self.U_params])
        return kron_sum(self.H, state, weights)

    def evolve(self, state: Array, dt: float, weights: dict):
        # Getting weights into same shape
        weights = jnp.concatenate([weights["rabi"] / 2, -weights["detune"], self.U_params])
        return second_order_trotter(self.H, state, dt, weights)


# Initializing Hamiltonian
n_qubits = 15
dt, N = 1e-3, 3000
laser_params = (1.0, 2.0)
U = jnp.triu(random.normal(key, (n_qubits, n_qubits)) ** 2)
in_state = init_state(n_qubits)


# We call it laser here but it's just a function which takes in 1) some parameters and 2) the time of the simulation
# and returns the parameter values of the hamiltonian. So it's really just a way to simulate time dependent hamiltonians.
def laser(laser_params, t):
    (w_rabi, w_detune) = laser_params
    return {
        "rabi": jnp.full((n_qubits,), 20.0 * jnp.cos(2 * jnp.pi * w_rabi * t)),
        "detune": jnp.full((n_qubits,), 15.0 * jnp.cos(2 * jnp.pi * w_detune * t)),
    }


hamiltonian = RydbergHamiltonian(n_qubits, U)
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
print("Note that for clarity we didn't jit the final function, so compilation time is included.")
