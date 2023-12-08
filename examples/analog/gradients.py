# This example shows how to calculate gradients
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from phaser import simulate
from phaser.models import RydbergHamiltonian
from phaser.utils import init_state

key = random.PRNGKey(42)

# Initializing Hamiltonian
n_qubits = 15
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


hamiltonian = RydbergHamiltonian(n_qubits, U)
hamiltonian_params = hamiltonian.init(
    key,
    in_state,
    laser(laser_params, 0),
)


# We take the gradient of some random state w.r.t the laser params and interaction_matrix
def forward(laser_params, hamiltonian_params):
    out_state = simulate(
        hamiltonian,
        hamiltonian_params,
        laser,
        laser_params,
        N,
        dt,
        in_state,
    )
    return (jnp.abs(out_state) ** 2).flatten()[-1]


# Getting the gradient fn w.r.t. both the pulse and interaction matrix and printing the grads
# Note that we jit (compile) the function so the timing here includes compiling
# but this only needs to happen once
grad_fn = jax.jit(jax.grad(forward, argnums=[0, 1]))
laser_grads, interaction_grads = grad_fn(laser_params, hamiltonian_params)

print(f"Gradients w.r.t laser params: \n {laser_grads}")
print(f"Gradients w.r.t interaction matrix: \n {interaction_grads}")
