from __future__ import annotations

from jax.config import config

config.update("jax_enable_x64", True)  # you should really really do this

import jax
import jax.numpy as jnp
from horqrux.gates import *
from horqrux.ops import apply_gate
from horqrux.utils import prepare_state


def circuit(state):
    # A list of gates in apply_gate denotes they're applied in parallel
    state = apply_gate(state, [H(0), NOT(1, 0), NOT(2, 0)])
    state = apply_gate(state, [Rx(1 / 4 * jnp.pi, 0), Ry(1 / 3 * jnp.pi, 1), Rz(1 / 2 * jnp.pi, 2)])
    state = apply_gate(state, [H(0), NOT(1, 0), NOT(2, 0)])
    state = apply_gate(state, [X(2), Y(1), Z(0)])
    return state


# defining initial state - don't do this inside the circuit as jit doesn't like it.
input_state = prepare_state(4, "0000")
circuit_fast = jax.jit(circuit)  # compiling for speed.

# Circuit is just a jax function so you can vmap, grad etc, multiple GPUs,
circuit(input_state)
