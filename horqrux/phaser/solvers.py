from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from chex import Array


@partial(jax.jit, static_argnames=("propagate_fn", "iterate_idx", "N"))
def forward_euler_solve(
    state: Array,
    propagate_fn: Callable,
    params: Any,
    N: int,
    dt: float,
    iterate_idx: bool = False,
) -> Array:
    def update_fn(state: Array, t: float) -> Array:
        return propagate_fn(params, state, t, dt), None

    # Iterate index gives step number instead of time
    # useful for comparing to pulser.

    if iterate_idx is True:
        t = jnp.arange(N)
    else:
        t = dt * jnp.arange(N)

    return jax.lax.scan(update_fn, state, t)[0]
