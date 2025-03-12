from __future__ import annotations

from collections import Counter
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from horqrux.composite import Observable, OpSequence
from horqrux.differentiation.ad import ad_expectation
from horqrux.differentiation.adjoint import adjoint_expectation as apply_adjoint
from horqrux.differentiation.gpsr import finite_shots_fwd, no_shots_fwd
from horqrux.utils import (
    DensityMatrix,
    DiffMode,
    State,
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
    n_shots: int = 0,
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
        n_shots (int): Number of shots. Defaults to 0 for no shots.
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
        if n_shots < 0:
            raise ValueError("The number of shots should be positive.")
        if n_shots == 0:
            return no_shots_fwd(
                state=state,
                gates=circuit.operations,
                observables=observables,
                values=values,
            )
        else:
            return finite_shots_fwd(
                state=state,
                gates=circuit.operations,
                observables=observables,
                values=values,
                n_shots=n_shots,
                key=key,
            )
