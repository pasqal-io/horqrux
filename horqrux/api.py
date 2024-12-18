from __future__ import annotations

from collections import Counter
from functools import singledispatch
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify

from horqrux.adjoint import adjoint_expectation
from horqrux.apply import apply_gate
from horqrux.primitive import GateSequence, Primitive
from horqrux.shots import finite_shots_fwd, observable_to_matrix
from horqrux.utils import (
    DensityMatrix,
    DiffMode,
    ForwardMode,
    OperationType,
    State,
    get_probas,
    inner,
    sample_from_probs,
)


def run(
    circuit: GateSequence,
    state: State,
    values: dict[str, float] = dict(),
) -> State:
    return apply_gate(state, circuit, values)


def sample(
    state: State,
    gates: GateSequence,
    values: dict[str, float] = dict(),
    n_shots: int = 1000,
) -> Counter:
    """Sample from a quantum program.

    Args:
        state (State): Input state vector or density matrix.
        gates (GateSequence): Sequence of gates.
        values (dict[str, float], optional): _description_. Defaults to dict().
        n_shots (int, optional): Parameter values.. Defaults to 1000.

    Raises:
        ValueError: If n_shots < 1.

    Returns:
        Counter: Bitstrings and frequencies.
    """
    if n_shots < 1:
        raise ValueError("You can only call sample with n_shots>0.")
    output_circuit = apply_gate(state, gates, values)

    if isinstance(output_circuit, DensityMatrix):
        n_qubits = len(output_circuit.array.shape) // 2
        d = 2**n_qubits
        output_circuit.array = output_circuit.array.reshape((d, d))
    else:
        n_qubits = len(output_circuit.array.shape)

    probs = get_probas(output_circuit)
    return sample_from_probs(probs, n_qubits, n_shots)


@singledispatch
def __ad_expectation_single_observable(
    state: Any,
    observable: Primitive,
    values: dict[str, float],
) -> Array:
    raise NotImplementedError("__ad_expectation_single_observable is not implemented")


@__ad_expectation_single_observable.register
def _(
    state: Array,
    observable: Primitive,
    values: dict[str, float],
) -> Array:
    projected_state = apply_gate(
        state,
        observable,
        values,
        OperationType.UNITARY,
    )
    return inner(state, projected_state).real


@__ad_expectation_single_observable.register
def _(
    state: DensityMatrix,
    observable: Primitive,
    values: dict[str, float],
) -> Array:
    n_qubits = len(state.array.shape) // 2
    mat_obs = observable_to_matrix(observable, n_qubits, values)
    d = 2**n_qubits
    prod = jnp.matmul(mat_obs, state.array.reshape((d, d)))
    return jnp.trace(prod, axis1=-2, axis2=-1).real


def ad_expectation(
    state: State,
    gates: GateSequence,
    observables: list[Primitive],
    values: dict[str, float],
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
       and compute the expectation given an observable.

    Args:
        state (State): Input state vector or density matrix.
        gates (GateSequence): Sequence of gates.
        observables (list[Primitive]): List of observables.
        values (dict[str, float]): Parameter values.

    Returns:
        Array: Expectation values.
    """
    outputs = [
        __ad_expectation_single_observable(
            apply_gate(state, gates, values, OperationType.UNITARY), observable, values
        )
        for observable in observables
    ]
    return jnp.stack(outputs)


def expectation(
    state: State,
    gates: GateSequence,
    observables: list[Primitive],
    values: dict[str, float],
    diff_mode: DiffMode = DiffMode.AD,
    forward_mode: ForwardMode = ForwardMode.EXACT,
    n_shots: Optional[int] = None,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.

    Args:
        state (State): Input state vector or density matrix.
        gates (GateSequence): Sequence of gates.
        observables (list[Primitive]): List of observables.
        values (dict[str, float]): Parameter values.
        diff_mode (DiffMode, optional): Differentiation mode. Defaults to DiffMode.AD.
        forward_mode (ForwardMode, optional): Type of forward method. Defaults to ForwardMode.EXACT.
        n_shots (Optional[int], optional): Number of shots. Defaults to None.
        key (Any, optional): Random key. Defaults to jax.random.PRNGKey(0).

    Returns:
        Array: Expectation values.
    """
    if diff_mode == DiffMode.AD:
        return ad_expectation(state, gates, observables, values)
    elif diff_mode == DiffMode.ADJOINT:
        if isinstance(state, DensityMatrix):
            raise ValueError("Adjoint does not support density matrices.")
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
        return finite_shots_fwd(
            state,
            gates,
            observables,
            values,
            n_shots=n_shots,
            key=key,
        )
