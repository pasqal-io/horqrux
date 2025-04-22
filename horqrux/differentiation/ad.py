from __future__ import annotations

from functools import singledispatch
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from horqrux.apply import apply_gates, apply_operator
from horqrux.composite import Observable, OpSequence
from horqrux.utils.operator_utils import (
    DensityMatrix,
    OperationType,
    State,
    inner,
    num_qubits,
)
from horqrux.utils.sparse_utils import real_sp, stack_sp


@singledispatch
def _ad_expectation_single_observable(
    state: Any,
    observable: Observable,
    values: dict[str, float],
    values_observable: dict[str, float] | None = None,
) -> Any:
    raise NotImplementedError("_ad_expectation_single_observable is not implemented")


@_ad_expectation_single_observable.register
def _(
    state: Array,
    observable: Observable,
    values: dict[str, float],
    values_observable: dict[str, float] | None = None,
) -> Array:
    values_observable = values_observable or values
    projected_state = observable.forward(
        state,
        values_observable,
    )
    return inner(state, projected_state).real


@_ad_expectation_single_observable.register
def _(
    state: BCOO,
    observable: Observable,
    values: dict[str, float],
    values_observable: dict[str, float] | None = None,
) -> Array:
    values_observable = values_observable or values
    projected_state = observable.forward(
        state,
        values_observable,
    )
    return real_sp(inner(state, projected_state))


@_ad_expectation_single_observable.register
def _(
    state: DensityMatrix,
    observable: Observable,
    values: dict[str, float],
    values_observable: dict[str, float] | None = None,
) -> Array:
    n_qubits = num_qubits(state)
    values_observable = values_observable or values
    mat_obs = observable.tensor(values_observable)
    d = 2**n_qubits
    prod = apply_operator(state.array, mat_obs, observable.qubit_support, (None,)).reshape((d, d))
    return jnp.trace(prod, axis1=-2, axis2=-1).real


def ad_expectation(
    state: State,
    circuit: OpSequence,
    observables: list[Observable],
    values: dict[str, float],
    values_observable: dict[str, float] | None = None,
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
       and compute the expectation given an observable.

    Args:
        state (State): Input state vector or density matrix.
        circuit (OpSequence): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.
        values_observable (dict[str, float], optional): Parameter values for the observable only.
            Useful for differentiation with respect to the observable parameters.
            Differentiation is only possible with DiffMode.AD.

    Returns:
        Array: Expectation values.
    """
    outputs = list(
        map(
            lambda observable: _ad_expectation_single_observable(
                apply_gates(state, list(iter(circuit)), values, OperationType.UNITARY),  # type: ignore[type-var]
                observable,
                values,
                values_observable,
            ),
            observables,
        )
    )
    return stack_sp(outputs)
