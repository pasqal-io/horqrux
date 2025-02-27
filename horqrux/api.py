from __future__ import annotations

from collections import Counter
from functools import singledispatch
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify

from horqrux.adjoint import adjoint_expectation as apply_adjoint
from horqrux.apply import apply_gates, apply_operator
from horqrux.composite import Observable, OpSequence
from horqrux.shots import finite_shots_fwd
from horqrux.utils import (
    DensityMatrix,
    DiffMode,
    ForwardMode,
    OperationType,
    State,
    inner,
    num_qubits,
    probabilities,
    sample_from_probs,
)


def run(
    circuit: OpSequence,
    state: State,
    values: dict[str, float] = dict(),
) -> State:
    return circuit(state, values)


def sample(
    state: State,
    circuit: OpSequence,
    values: dict[str, float] = dict(),
    n_shots: int = 1000,
) -> Counter:
    """Sample from a quantum program.

    Args:
        state (State): Input state vector or density matrix.
        circuit (OpSequence): Sequence of gates.
        values (dict[str, float], optional): _description_. Defaults to dict().
        n_shots (int, optional): Parameter values.. Defaults to 1000.

    Raises:
        ValueError: If n_shots < 1.

    Returns:
        Counter: Bitstrings and frequencies.
    """
    if n_shots < 1:
        raise ValueError("You can only sample with non-negative 'n_shots'.")
    output_circuit = circuit(state, values)
    n_qubits = num_qubits(output_circuit)
    if isinstance(output_circuit, DensityMatrix):
        d = 2**n_qubits
        output_circuit.array = output_circuit.array.reshape((d, d))

    probs = probabilities(output_circuit)
    return sample_from_probs(probs, n_qubits, n_shots)


@singledispatch
def _ad_expectation_single_observable(
    state: Any,
    observable: Observable,
    values: dict[str, float],
) -> Any:
    raise NotImplementedError("_ad_expectation_single_observable is not implemented")


@_ad_expectation_single_observable.register
def _(
    state: Array,
    observable: Observable,
    values: dict[str, float],
) -> Array:
    projected_state = observable(
        state,
        values,
    )
    return inner(state, projected_state).real


@_ad_expectation_single_observable.register
def _(
    state: DensityMatrix,
    observable: Observable,
    values: dict[str, float],
) -> Array:
    n_qubits = num_qubits(state)
    mat_obs = observable.tensor(values)
    d = 2**n_qubits
    prod = apply_operator(state.array, mat_obs, observable.qubit_support, (None,)).reshape((d, d))
    return jnp.trace(prod, axis1=-2, axis2=-1).real


def ad_expectation(
    state: State,
    circuit: OpSequence,
    observables: list[Observable],
    values: dict[str, float],
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
       and compute the expectation given an observable.

    Args:
        state (State): Input state vector or density matrix.
        circuit (OpSequence): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.

    Returns:
        Array: Expectation values.
    """
    outputs = list(
        map(
            lambda observable: _ad_expectation_single_observable(
                apply_gates(state, circuit.operations, values, OperationType.UNITARY),
                observable,
                values,
            ),
            observables,
        )
    )
    return jnp.stack(outputs)


def adjoint_expectation(
    state: State,
    circuit: OpSequence,
    observables: list[Observable],
    values: dict[str, float],
) -> Array:
    """Apply a sequence of adjoint operators to an input state given parameters 'values'
       and compute the expectation given an observable.

    Args:
        state (State): Input state vector or density matrix.
        circuit (OpSequence): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.

    Returns:
        Array: Expectation values.
    """
    outputs = list(
        map(
            lambda observable: apply_adjoint(state, circuit.operations, observable, values),
            observables,
        )
    )
    return jnp.stack(outputs)


def expectation(
    state: State,
    circuit: OpSequence,
    observables: list[Observable],
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
        circuit (OpSequence): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.
        diff_mode (DiffMode, optional): Differentiation mode. Defaults to DiffMode.AD.
        forward_mode (ForwardMode, optional): Type of forward method. Defaults to ForwardMode.EXACT.
        n_shots (Optional[int], optional): Number of shots. Defaults to None.
        key (Any, optional): Random key. Defaults to jax.random.PRNGKey(0).

    Returns:
        Array: Expectation values.
    """
    if diff_mode == DiffMode.AD:
        return ad_expectation(state, circuit, observables, values)
    elif diff_mode == DiffMode.ADJOINT:
        if isinstance(state, DensityMatrix):
            raise TypeError("Adjoint does not support density matrices.")
        return adjoint_expectation(state, circuit, observables, values)
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
            state=state,
            gates=circuit.operations,
            observables=observables,
            values=values,
            n_shots=n_shots,
            key=key,
        )
