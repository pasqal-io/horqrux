from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
from jax import Array

from horqrux.adjoint import adjoint_expectation
from horqrux.apply import apply_gate
from horqrux.primitive import Primitive
from horqrux.utils import DiffMode, ForwardMode, OperationType, inner


def run(
    circuit: list[Primitive],
    state: Array,
    values: dict[str, float] = dict(),
) -> Array:
    return apply_gate(state, circuit, values)


def sample(
    state: Array,
    gates: list[Primitive],
    values: dict[str, float] = dict(),
    n_shots: int = 1000,
) -> Counter:
    if n_shots < 1:
        raise ValueError("You can only call sample with n_shots>0.")

    wf = apply_gate(state, gates, values)
    probs = jnp.abs(jnp.float_power(wf, 2.0)).ravel()
    key = jax.random.PRNGKey(0)
    n_qubits = len(state.shape)
    # JAX handles pseudo random number generation by tracking an explicit state via a random key
    # For more details, see https://jax.readthedocs.io/en/latest/random-numbers.html
    samples = jax.vmap(
        lambda subkey: jax.random.choice(key=subkey, a=jnp.arange(0, 2**n_qubits), p=probs)
    )(jax.random.split(key, n_shots))

    return Counter(
        {
            format(k, "0{}b".format(n_qubits)): count.item()
            for k, count in enumerate(jnp.bincount(samples))
            if count > 0
        }
    )


def ad_expectation(
    state: Array, gates: list[Primitive], observable: list[Primitive], values: dict[str, float]
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return inner(out_state, projected_state).real


def expectation(
    state: Array,
    gates: list[Primitive],
    observable: list[Primitive],
    values: dict[str, float],
    diff_mode: DiffMode = DiffMode.AD,
    forward_mode: ForwardMode = ForwardMode.EXACT,
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    if diff_mode == DiffMode.AD:
        return ad_expectation(state, gates, observable, values)
    elif diff_mode == DiffMode.ADJOINT:
        return adjoint_expectation(state, gates, observable, values)
    elif diff_mode == DiffMode.GPSR:
        assert forward_mode == ForwardMode.SHOTS
        return NotImplementedError()
