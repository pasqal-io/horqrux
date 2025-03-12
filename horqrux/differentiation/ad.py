from __future__ import annotations

from functools import singledispatch
from typing import Any, Union

import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from horqrux.apply import apply_gates, apply_operator
from horqrux.composite import Observable, OpSequence
from horqrux.sparse_utils import real_sp, stack_sp
from horqrux.utils import (
    DensityMatrix,
    OperationType,
    State,
    inner,
    num_qubits,
)


@singledispatch
def _ad_expectation_single_observable(
    state: Any,
    observable: Observable,
    values: dict[str, float],
) -> Any:
    raise NotImplementedError("_ad_expectation_single_observable is not implemented")


@_ad_expectation_single_observable.register
def _(
    state: Union[Array, BCOO],
    observable: Observable,
    values: dict[str, float],
) -> Array:
    projected_state = observable(
        state,
        values,
    )
    return real_sp(inner(state, projected_state))


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
    return stack_sp(outputs)
