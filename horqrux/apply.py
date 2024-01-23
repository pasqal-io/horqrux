from __future__ import annotations

from functools import reduce
from operator import add
from typing import Iterable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from horqrux.abstract import Operator

from .utils import ControlQubits, State, TargetQubits, _controlled, is_controlled


def apply_operator(
    state: State,
    operator: Array,
    target: TargetQubits,
    control: ControlQubits,
) -> State:
    """Applies a single or series of operators to the given state.
       The operators are expected to either be an array over whose first axis we can iterate (e.g. [N_gates, 2 x 2])
       or if you have a mix of single and multi qubit gates a tuple or list like [O_1, O_2, ...].
       This function then sequentially applies this gates, adding control bits
       as necessary and returning the state after applying all the gates.


    Args:
        state: Input state to operate on.
        operator: List of arrays or array of operators to contract over the state.
        target: Target indices, Tuple of Tuple of ints.
        control: Control indices, Tuple of length target_idex of None or Tuple.

    Returns:
        Array: Changed state.
    """
    qubits = target
    assert isinstance(control, tuple)
    if is_controlled(control):
        operator = _controlled(operator, len(control))
        qubits = (*control, *target)
    n_qubits = int(np.log2(operator.size))
    operator = operator.reshape(tuple(2 for _ in np.arange(n_qubits)))
    op_dims = tuple(np.arange(operator.ndim // 2, operator.ndim, dtype=int))
    state = jnp.tensordot(a=operator, b=state, axes=(op_dims, qubits))
    return jnp.moveaxis(a=state, source=np.arange(len(qubits)), destination=qubits)


def apply_gate(
    state: State, gate: Operator | Iterable[Operator], values: dict[str, float] = {}
) -> State:
    """Applies a gate to given state. Essentially a simple wrapper around
       apply_operator, see that docstring for more info.

    Arguments:
        state (Array): State to operate on.
        gate (Gate): Gate(s) to apply.

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
