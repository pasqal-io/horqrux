from __future__ import annotations

from functools import reduce
from operator import add
from typing import Iterable

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
    # TODO: Not pretty but works for now.
    if isinstance(gate, list) | isinstance(gate, tuple):
        O = [g.O for g in gate]  # type: ignore[union-attr]
        target_idx = reduce(add, [g.target_idx for g in gate])  # type: ignore[union-attr]
        control_idx = reduce(add, [g.control_idx for g in gate])  # type: ignore[union-attr]
    else:
        O, target_idx, control_idx = (gate.O,), gate.target_idx, gate.control_idx  # type: ignore[assignment,union-attr]

    return reduce(
        lambda state, inputs: apply_operator(state, *inputs),
        zip(O, target_idx, control_idx),
        state,
    )


def apply_operator(
    state: State,
    O: ArrayLike,
    target_idx: TargetIdx,
    control_idx: ControlIdx,
) -> State:
    """Applies a single or series of operators to the given state. The operators O should
       either be an array over whose first axis we can iterate (e.g. [N_gates, 2 x 2])
       or if you have a mix of single and multi qubit gates a tuple or list like [O_1, O_2, ...].
       This function then sequentially applies this gates, adding control bits
       as necessary and returning the state after applying all the gates.


    Args:
        state (Array): Input state to operate on.
        O (Union[Iterable, Array]): Iterable or array of operator matrixes to apply.
        target_idx (TargetIdx): Target indices, Tuple of Tuple of ints.
        control_idx (ControlIdx): Control indices, Tuple of length target_idex of None or Tuple.

    Returns:
        Array: Changed state.
    """

    def make_controlled(O: Array, n_control: int) -> Array:
        n_qubit_gate = int(np.log2(O.shape[0]))
        O_c = jnp.eye(2 ** (n_control + n_qubit_gate))
        O_c = O_c.at[-(2**n_qubit_gate) :, -(2**n_qubit_gate) :].set(O)
        return hilbert_reshape(O_c)

    if control_idx is not None:
        O = make_controlled(O, len(control_idx))
        target_idx = (*control_idx, *target_idx)  # type: ignore[arg-type]

    if len(target_idx) > 1:
        O = hilbert_reshape(O)

    op_dims = tuple(np.arange(O.ndim // 2, O.ndim, dtype=int))
    new_state = jnp.tensordot(O, state, axes=(op_dims, target_idx))
    return jnp.moveaxis(new_state, np.arange(len(target_idx)), target_idx)
