from __future__ import annotations

from functools import reduce
from operator import add
from typing import Iterable, Tuple

import jax.numpy as jnp
import numpy as np

from .abstract import Operator
from .matrices import make_controlled
from .utils import ConcretizedOperator, QubitSupport, State, hilbert_reshape


def apply_operator(
    state: State,
    unitary: ConcretizedOperator,
    target: QubitSupport,
    control: QubitSupport,
) -> State:
    """Applies a single or series of operators to the given state. The operators 'operator' should
       either be an array over whose first axis we can iterate (e.g. [N_gates, 2 x 2])
       or if you have a mix of single and multi qubit gates a tuple or list like [O_1, O_2, ...].
       This function then sequentially applies this gates, adding control bits
       as necessary and returning the state after applying all the gates.


    Args:
        state (Array): Input state to operate on.
        operator (Union[Iterable, Array]): Iterable or array of operator matrixes to apply.
        target_idx (TargetIdx): Target indices, Tuple of Tuple of ints.
        control_idx (ControlIdx): Control indices, Tuple of length target_idex of None or Tuple.

    Returns:
        Array: Changed state.
    """
    qubits = target
    if None not in control:
        unitary = make_controlled(unitary, len(control))
        qubits = control + target
    n_support = len(qubits)
    unitary = hilbert_reshape(unitary) if len(target) > 1 else unitary
    op_dims = tuple(range(n_support, 2 * n_support))
    state = jnp.tensordot(a=unitary, b=state, axes=(op_dims, qubits))
    return jnp.moveaxis(a=state, source=np.arange(n_support), destination=qubits)


def apply_gate(state: State, gate: Operator | Iterable[Operator]) -> State:
    """Applies gate to given state. Essentially a simple wrapper around
       apply_operator, see that docstring for more info.

    Args:
        state (Array): State to operate on.
        gate (Gate): Gate(s) to apply.

    Returns:
        Array: Changed state.
    """
    unitary: Tuple[ConcretizedOperator, ...]
    if isinstance(gate, Operator):
        unitary, target, control = (gate.unitary,), gate.target, gate.control
    else:
        unitary = tuple(g.unitary for g in gate)
        target = reduce(add, [g.target for g in gate])
        control = reduce(add, [g.control for g in gate])
    return reduce(
        lambda state, gate: apply_operator(state, *gate),
        zip(unitary, (target,), (control,)),
        state,
    )
