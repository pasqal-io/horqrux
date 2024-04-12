from __future__ import annotations

from functools import reduce
from operator import add
from typing import Iterable, Tuple, no_type_check

import jax.numpy as jnp
import numpy as np
from jax import Array

from horqrux.primitive import Primitive

from .utils import OperationType, State, _controlled, is_controlled


def apply_operator(
    state: State,
    operator: Array,
    target: Tuple[int, ...],
    control: Tuple[int | None, ...],
) -> State:
    """Applies an operator, i.e. a single array of shape [2, 2, ...], on a given state
       of shape [2 for _ in range(n_qubits)] for a given set of target and control qubits.
       In case of a controlled operation, the 'operator' array will be embedded into a controlled array.

       Since dimension 'i' in 'state' corresponds to all amplitudes where qubit 'i' is 1,
       target and control qubits represent the dimensions over which to contract the 'operator'.
       Contraction means applying the 'dot' operation between the operator array and dimension 'i'
       of 'state, resulting in a new state where the result of the 'dot' operation has been moved to
       dimension 'i' of 'state'. To restore the former order of dimensions, the affected dimensions
       are moved to their original positions and the state is returned.

    Arguments:
        state: State to operate on.
        operator: Array to contract over 'state'.
        target: Tuple of target qubits on which to apply the 'operator' to.
        control: Tuple of control qubits.

    Returns:
        State after applying 'operator'.
    """
    state_dims: Tuple[int, ...] = target
    if is_controlled(control):
        operator = _controlled(operator, len(control))
        state_dims = (*control, *target)  # type: ignore[arg-type]
    n_qubits = int(np.log2(operator.size))
    operator = operator.reshape(tuple(2 for _ in np.arange(n_qubits)))
    op_dims = tuple(np.arange(operator.ndim // 2, operator.ndim, dtype=int))
    state = jnp.tensordot(a=operator, b=state, axes=(op_dims, state_dims))
    new_state_dims = tuple(i for i in range(len(state_dims)))
    return jnp.moveaxis(a=state, source=new_state_dims, destination=state_dims)


@no_type_check
def merge_operators(
    operators: tuple[Array, ...], targets: tuple[int], controls: tuple[int]
) -> tuple[tuple[Array, ...], tuple[int, ...], tuple[int, ...]]:
    if len(operators) == 1:
        return operators, targets, controls
    operators = operators[::-1]
    targets = targets[::-1]
    controls = controls[::-1]
    merged_operator = operators[0]
    merged_target = targets[0]
    merged_control = controls[0]
    new_operators = tuple()
    new_targets = tuple()
    new_controls = tuple()
    for op, target, control in zip(operators[1:], targets[1:], controls[1:]):
        if target == merged_target and control == merged_control:
            merged_operator = merged_operator @ op
        else:
            new_operators = new_operators + (merged_operator,)
            new_targets = new_targets + (merged_target,)
            new_controls = new_controls + (merged_control,)
            merged_operator = op
            merged_target = target
            merged_control = control
    if merged_operator is not None:
        new_operators = new_operators + (merged_operator,)
        new_targets = new_targets + (merged_target,)
        new_controls = new_controls + (merged_control,)
    return new_operators[::-1], new_targets[::-1], new_controls[::-1]


def apply_gate(
    state: State,
    gate: Primitive | Iterable[Primitive],
    values: dict[str, float] = dict(),
    op_type: OperationType = OperationType.UNITARY,
) -> State:
    """Wrapper function for 'apply_operator' which applies a gate or a series of gates to a given state.
    Arguments:
        state: State to operate on.
        gate: Gate(s) to apply.
        values: A dictionary with parameter values.
        op_type: The type of operation to perform: Unitary, Dagger or Jacobian.

    Returns:
        State after applying 'gate'.
    """
    operator: Tuple[Array, ...]
    if isinstance(gate, Primitive):
        operator_fn = getattr(gate, op_type)
        operator, target, control = (operator_fn(values),), gate.target, gate.control
    else:
        operator = tuple(getattr(g, op_type)(values) for g in gate)
        target = reduce(add, [g.target for g in gate])
        control = reduce(add, [g.control for g in gate])
        operator, target, control = merge_operators(operator, target, control)
    return reduce(
        lambda state, gate: apply_operator(state, *gate),
        zip(operator, target, control),
        state,
    )
