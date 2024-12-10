from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify

from horqrux.adjoint import adjoint_expectation
from horqrux.apply import apply_gate
from horqrux.primitive import GateSequence, Primitive
from horqrux.shots import finite_shots_fwd
from horqrux.utils import DiffMode, ForwardMode, OperationType, inner


def run(
    circuit: GateSequence,
    state: Array,
    values: dict[str, float] = dict(),
    is_state_densitymat: bool = False,
) -> Array:
    return apply_gate(state, circuit, values, is_state_densitymat=is_state_densitymat)


def sample(
    state: Array,
    gates: GateSequence,
    values: dict[str, float] = dict(),
    n_shots: int = 1000,
    is_state_densitymat: bool = False,
) -> Counter:
    if n_shots < 1:
        raise ValueError("You can only call sample with n_shots>0.")

    if is_state_densitymat:
        raise NotImplementedError("Sampling with density matrices is not yet supported!")

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


def __ad_expectation_single_observable(
    state: Array,
    gates: GateSequence,
    observable: Primitive,
    values: dict[str, float],
    is_state_densitymat: bool = False,
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    if not is_state_densitymat:
        return inner(out_state, projected_state).real

    raise NotImplementedError("Expectation from density matrices is not yet supported!")


def ad_expectation(
    state: Array,
    gates: GateSequence,
    observables: list[Primitive],
    values: dict[str, float],
    is_state_densitymat: bool = False,
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    outputs = [
        __ad_expectation_single_observable(state, gates, observable, values, is_state_densitymat)
        for observable in observables
    ]
    return jnp.stack(outputs)


def expectation(
    state: Array,
    gates: GateSequence,
    observables: list[Primitive],
    values: dict[str, float],
    diff_mode: DiffMode = DiffMode.AD,
    forward_mode: ForwardMode = ForwardMode.EXACT,
    n_shots: Optional[int] = None,
    is_state_densitymat: bool = False,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    if diff_mode == DiffMode.AD:
        return ad_expectation(state, gates, observables, values)
    elif diff_mode == DiffMode.ADJOINT:
        return adjoint_expectation(state, gates, observables, values)
    elif diff_mode == DiffMode.GPSR:
        checkify.check(
            forward_mode == ForwardMode.SHOTS, "Finite shots and GPSR must be used together"
        )
        checkify.check(
            type(n_shots) is int,
            "Number of shots must be an integer for finite shots.",
        )
        # Type checking is disabled because mypy doesn't parse checkify.check.
        # type: ignore
        # if is_state_densitymat:
        #     raise NotImplementedError("Expectation with density matrices is not yet supported!")
        return finite_shots_fwd(
            state,
            gates,
            observables,
            values,
            n_shots=n_shots,
            is_state_densitymat=is_state_densitymat,
            key=key,
        )
