from __future__ import annotations

import jax.numpy as jnp


def state_overlap(state_A, state_B):
    return jnp.abs(jnp.dot(jnp.conjugate(state_A.flatten()), state_B.flatten()))


def state_norm(state):
    return jnp.sum(jnp.abs(state) ** 2)
