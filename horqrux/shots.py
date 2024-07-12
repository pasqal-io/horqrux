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

    # eigvals: an array of shape (..., M) containing the eigenvalues.
    # eigvecs: an array of shape (..., M, M), where column v[:, i]
    # is the eigenvector corresponding to the eigenvalue w[i].

    eigvec_state_prod = jnp.multiply(eigvecs.flatten(), jnp.conjugate(state.T).flatten())
    probs = jnp.abs(jnp.float_power(eigvec_state_prod, 2.0)).ravel()
    key = jax.random.PRNGKey(0)
    n_qubits = len(state.shape)
    samples = jax.vmap(
        lambda subkey: jax.random.choice(key=subkey, a=jnp.arange(0, 2**n_qubits), p=probs)
    )(jax.random.split(key, n_shots))
    normalized_samples = jnp.bincount(samples) / n_shots
    # change this einsum to be generic regading the number of dims
    return jnp.einsum("i,ji ->", eigvals, normalized_samples.reshape([2] * n_qubits)).real
