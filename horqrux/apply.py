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
    unitary: Array,
    target: Tuple[int, ...],
    control: Tuple[int | None, ...],
) -> State:
    """Applies a unitary, i.e. a single array of shape [2, 2, ...], on a given state
       of shape [2 for _ in range(n_qubits)] for a given set of target and control qubits.
       In case of control qubits, the 'unitary' array will be embedded into a controlled array.

       Since dimension 'i' in 'state' corresponds to all amplitudes which are affected by qubit 'i',
       target and control qubits correspond to dimensions to contract 'unitary' over.
       Contraction over qubit 'i' means applying the 'dot' operation between 'unitary' and dimension 'i'
       of 'state, resulting in a new state where the result of the 'dot' operation has been moved to
       dimension 'i' of 'state'. To restore the former order of dimensions, the affected dimensions
       are moved to their original positions and the state is returned.

    Arguments:
        state: State to operate on.
        unitary: Array to contract over 'state'.
        target: Tuple of target qubits on which to apply the 'operator' to.
        control: Tuple of control qubits.

    Returns:
        State after applying 'unitary'.
    """
    state_dims: Tuple[int, ...] = target
    if is_controlled(control):
        unitary = _controlled(unitary, len(control))
        state_dims = (*control, *target)  # type: ignore[arg-type]
    n_qubits = int(np.log2(unitary.size))
    unitary = unitary.reshape(tuple(2 for _ in np.arange(n_qubits)))
    op_dims = tuple(np.arange(unitary.ndim // 2, unitary.ndim, dtype=int))
    state = jnp.tensordot(a=unitary, b=state, axes=(op_dims, state_dims))
    new_state_dims = tuple(i for i in range(len(state_dims)))
    return jnp.moveaxis(a=state, source=new_state_dims, destination=state_dims)


def apply_gate(
    state: State, gate: Operator | Iterable[Operator], values: dict[str, float] = dict()
) -> State:
    """Wrapper function for 'apply_operator' which applies a gate or a series of gates to a given state.
    Arguments:
        state: State to operate on.
        gate: Gate(s) to apply.

    Returns:
        State after applying 'gate'.
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
