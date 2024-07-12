from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import Array, vmap

from horqrux.apply import apply_gate
from horqrux.primitive import Primitive


def finite_shots_fwd(
    state: Array,
    gates: list[Primitive],
    observable: Primitive,
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
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
    normalized_samples = jnp.bincount(samples, length=state.size) / n_shots
    # change this einsum to be generic regading the number of dims
    return jnp.einsum("i,ji ->", eigvals, normalized_samples.reshape([2] * n_qubits)).real, (
        state,
        gates,
        observable,
        values,
        n_shots,
        key,
    )


@jax.custom_vjp
def finite_shots(
    state: Array,
    gates: list[Primitive],
    observable: Primitive,
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
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
    normalized_samples = jnp.bincount(samples, length=state.size) / n_shots
    # change this einsum to be generic regading the number of dims

    return jnp.einsum("i,ji ->", eigvals, normalized_samples.reshape([2] * n_qubits)).real


def finite_shots_vjp(primals: Tuple, tangents: Array) -> Array:
    state, gates, observable, values, n_shots, key = primals
    # (tangents,) = tangents

    def expfn(values: dict) -> Array:
        return finite_shots(state, gates, observable, values, n_shots, key)

    spectral_gap = 2.0
    # NOTE compute spectral gap through the generator which is associated with a param_name
    shift = jnp.pi / 2
    grads = {key: None for key in values.keys()}

    def shift_circ(param_name: str, values: dict) -> Array:
        shifted_values = values.copy()
        shiftvals = jnp.array(
            [shifted_values[param_name] + shift, shifted_values[param_name] - shift]
        )

        def _expectation(val: Array) -> Array:
            shifted_values[param_name] = val
            return expfn(shifted_values)

        return vmap(_expectation, in_axes=(0,))(shiftvals)

    for param_name in values.keys():
        f_plus, f_min = shift_circ(param_name, values)
        grad = spectral_gap * (f_plus - f_min) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        grads[param_name] = tangents * grad
    return (None, None, None, grads, None, None)


finite_shots.defvjp(finite_shots_fwd, finite_shots_vjp)
