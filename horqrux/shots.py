from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from horqrux.apply import apply_gate
from horqrux.primitive import Primitive


def finite_shots(
    state: Array,
    gates: list[Primitive],
    observable: Primitive,
    values: dict[str, float],
    diff_mode: str = "gpsr",
    n_shots: int = 100,
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    state = apply_gate(state, gates, values)
    # NOTE this only works now for an observable comprised of a single gate
    # to get eigvals,eigvecs for arbitary compositions of paulis, we need to
    # create the full tensor. check `block_to_jax` in qadence for this
    eigvals, eigvecs = jnp.linalg.eig(observable.unitary())
    probs = jnp.abs(jnp.float_power(jnp.inner(state, eigvecs), 2.0)).ravel()
    key = jax.random.PRNGKey(0)
    n_qubits = len(state.shape)
    samples = jax.vmap(
        lambda subkey: jax.random.choice(key=subkey, a=jnp.arange(0, 2**n_qubits), p=probs)
    )(jax.random.split(key, n_shots))
    # samples now contains a list of indices
    # i forgot the formula
    # something here which is correct
    counts = jnp.bincount(samples)
    return jnp.mean(counts / n_shots)
