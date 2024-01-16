from __future__ import annotations

from functools import reduce
from operator import add
from typing import Iterable

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .matrices import make_controlled
from .types import ControlIdx, Gate, State, TargetIdx
from .utils import hilbert_reshape


def apply_gate(state: State, gate: Gate | Iterable[Gate]) -> State:
    """Applies gate to given state. Essentially a simple wrapper around
    apply_operator, see that docstring for more info.

    Args:
        state (Array): State to operate on.
        gate (Gate): Gate(s) to apply.

    Returns:
        Array: Changed state.
    """
    if isinstance(gate, list) | isinstance(gate, tuple):
        op = [g.operator for g in gate]  # type: ignore[union-attr]
        target = reduce(add, [g.target_idx for g in gate])  # type: ignore[union-attr]
        control = reduce(add, [g.control_idx for g in gate])  # type: ignore[union-attr]
    else:
        op, target, control = (gate.operator,), gate.target_idx, gate.control_idx  # type: ignore[assignment,union-attr]

    return reduce(
        lambda state, inputs: apply_operator(state, *inputs),
        zip(op, target, control),
        state,
    )


def apply_operator(
    state: State,
    operator: ArrayLike,
    target_idx: TargetIdx,
    control_idx: ControlIdx,
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

    if control_idx is not None:
        operator = make_controlled(operator, len(control_idx))
        target_idx = (*control_idx, *target_idx)  # type: ignore[arg-type]

    if len(target_idx) > 1:
        operator = hilbert_reshape(operator)

    op_dims = tuple(np.arange(operator.ndim // 2, operator.ndim, dtype=int))
    new_state = jnp.tensordot(operator, state, axes=(op_dims, target_idx))
    return jnp.moveaxis(new_state, np.arange(len(target_idx)), target_idx)
