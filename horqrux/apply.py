from __future__ import annotations

from functools import reduce
from operator import add
from typing import Iterable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from horqrux.abstract import Operator

from .utils import State, _controlled, is_controlled


def apply_operator(
    state: State,
    operator: Array,
    target: Tuple[int, ...],
    control: Tuple[int | None, ...],
) -> State:
    """Applies a single array corresponding to an operator to a given state
       for a given set of target and control qubits.

    Args:
        state: State to operate on.
        operator: Array to contract over 'state'.
        target: Tuple of target qubits on which to apply the 'operator' to.
        control: Tuple of control qubits.

    Returns:
        State after applying 'operator'.
    """
    qubits: Tuple[int, ...] = target
    if is_controlled(control):
        operator = _controlled(operator, len(control))
        qubits = (*control, *target)
    n_qubits = int(np.log2(operator.size))
    operator = operator.reshape(tuple(2 for _ in np.arange(n_qubits)))
    op_dims = tuple(np.arange(operator.ndim // 2, operator.ndim, dtype=int))
    state = jnp.tensordot(a=operator, b=state, axes=(op_dims, qubits))
    new_dims = tuple(i for i in range(len(qubits)))
    return jnp.moveaxis(a=state, source=new_dims, destination=qubits)


def apply_gate(
    state: State, gate: Operator | Iterable[Operator], values: dict[str, float] = {}
) -> State:
    """Applies a gate or a series of gates to a given state.
       This function sequentially applies 'gate', adding control bits
       as necessary and returning the state after applying all the gates.

    Arguments:
        state: State to operate on.
        gate: Gate(s) to apply.

    Returns:
        Array: Changed state.
    """
    unitary: Tuple[Array, ...]
    if isinstance(gate, Operator):
        unitary, target, control = (gate.unitary(values),), gate.target, gate.control
    else:
        unitary = tuple(g.unitary(values) for g in gate)
        target = reduce(add, [g.target for g in gate])
        control = reduce(add, [g.control for g in gate])
    return reduce(
        lambda state, gate: apply_operator(state, *gate),
        zip(unitary, target, control),
        state,
    )
