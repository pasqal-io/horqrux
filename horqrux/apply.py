from __future__ import annotations

from functools import partial, reduce, singledispatch
from operator import add
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse import BCOO, sparsify

from horqrux.noise import DigitalNoiseInstance, NoiseProtocol
from horqrux.primitives.primitive import Primitive
from horqrux.utils.operator_utils import (
    DensityMatrix,
    OperationType,
    State,
    _controlled,
    _dagger,
    density_mat,
    is_controlled,
    permute_basis,
)


@singledispatch
def apply_operator(
    state: Any,
    operator: Array,
    target: tuple[int, ...],
    control: tuple[Union[int, None], ...],
) -> Any:
    """Apply an operator on a state or density matrix.

    Args:
        state (Any): Array to operate on.
        operator (Array): Array to contract over 'state'.
        target (tuple[int, ...]): tuple of target qubits on which to apply the 'operator' to.
        control (tuple[int  |  None, ...]): tuple of control qubits.

    Raises:
        NotImplementedError: If not implemented for given types.

    Returns:
        Array: The output of the application of the operator.
    """
    raise NotImplementedError("apply_operator is not implemented")


@apply_operator.register
def _(
    state: Array,
    operator: Array,
    target: tuple[int, ...],
    control: tuple[Union[int, None], ...],
) -> Array:
    """Applies an operator, i.e. a single array of shape [2, 2, ...], on a given state
       of shape [2 for _ in range(n_qubits)] for a given set of target and control qubits.
       In case of a controlled operation, the 'operator' array will be embedded into a controlled array.

       Since dimension 'i' in 'state' corresponds to all amplitudes where qubit 'i' is 1,
       target and control qubits represent the dimensions over which to contract the 'operator'.
       Contraction means applying the 'dot' operation between the operator array and dimension 'i'
       of 'state, resulting in a new state where the result of the 'dot' operation has been moved to
       dimension 'i' of 'state'. To restore the former order of dimensions, the affected dimensions
       are moved to their original positions and the state is returned.

    Args:
        state (Array): Array to operate on.
        operator (Array): Array to contract over 'state'.
        target (tuple[int, ...]): tuple of target qubits on which to apply the 'operator' to.
        control (tuple[int  |  None, ...]): tuple of control qubits.

    Returns:
        Array after applying 'operator'.
    """
    state_dims: tuple[int, ...] = target
    if is_controlled(control):
        operator = _controlled(operator, len(control))
        state_dims = (*control, *target)  # type: ignore[arg-type]
    n_qubits_op = int(np.log2(operator.shape[1]))
    operator = operator.reshape((2,) * (2 * n_qubits_op))
    op_out_dims = tuple(np.arange(operator.ndim // 2, operator.ndim, dtype=int))
    # Apply operator
    new_state_dims = tuple(range(len(state_dims)))
    state = jnp.tensordot(a=operator, b=state, axes=(op_out_dims, state_dims))
    return jnp.moveaxis(a=state, source=new_state_dims, destination=state_dims)


@apply_operator.register
def _(
    state: BCOO,
    operator: BCOO,
    target: tuple[int, ...],
    control: tuple[Union[int, None], ...],
) -> Array:
    """Applies an operator, i.e. a single array of shape [2, 2, ...], on a given state
       of shape [2] * n_qubits for a given set of target and control qubits.
       In case of a controlled operation, the 'operator' array will be embedded into a controlled array.

       Since dimension 'i' in 'state' corresponds to all amplitudes where qubit 'i' is 1,
       target and control qubits represent the dimensions over which to contract the 'operator'.
       Contraction means applying the 'dot' operation between the operator array and dimension 'i'
       of 'state, resulting in a new state where the result of the 'dot' operation has been moved to
       dimension 'i' of 'state'. To restore the former order of dimensions, the affected dimensions
       are moved to their original positions and the state is returned.

    Args:
        state (Array): Array to operate on.
        operator (Array): Array to contract over 'state'.
        target (tuple[int, ...]): tuple of target qubits on which to apply the 'operator' to.
        control (tuple[int  |  None, ...]): tuple of control qubits.

    Returns:
        Array after applying 'operator'.
    """
    state_dims: tuple[int, ...] = target
    if is_controlled(control):
        operator = _controlled(operator, len(control))
        state_dims = (*control, *target)  # type: ignore[arg-type]
    n_qubits_op = int(np.log2(operator.shape[1]))
    operator = operator.reshape((2,) * (2 * n_qubits_op))
    op_out_dims = tuple(np.arange(operator.ndim // 2, operator.ndim, dtype=int))
    # Apply operator
    new_state_dims = tuple(range(len(state_dims)))
    tensordot_sp = sparsify(lambda a, b: jnp.tensordot(a=a, b=b, axes=(op_out_dims, state_dims)))
    moveaxis_sp = sparsify(
        lambda a: jnp.moveaxis(a=a, source=new_state_dims, destination=state_dims)
    )
    state = tensordot_sp(operator, state)
    return moveaxis_sp(state)


@apply_operator.register
def _(
    state: DensityMatrix,
    operator: Array,
    target: tuple[int, ...],
    control: tuple[Union[int, None], ...],
) -> DensityMatrix:
    """Applies an operator, i.e. a single array of shape [2, 2, ...], on a given density matrix
       of shape [2 for _ in range(2 * n_qubits)] for a given set of target and control qubits.
       In case of a controlled operation, the 'operator' array will be embedded into a controlled array.

    Args:
        state (DensityMatrix): Array to operate on.
        operator (Array): Array to contract over 'state'.
        target (tuple[int, ...]): tuple of target qubits on which to apply the 'operator' to.
        control (tuple[int  |  None, ...]): tuple of control qubits.

    Returns:
        Density matrix after applying 'operator'.
    """
    state_dims: tuple[int, ...] = target
    if is_controlled(control):
        operator = _controlled(operator, len(control))
        state_dims = (*control, *target)  # type: ignore[arg-type]
    n_qubits_op = int(np.log2(operator.shape[1]))
    operator = operator.reshape((2,) * (2 * n_qubits_op))
    op_out_dims = tuple(np.arange(operator.ndim // 2, operator.ndim, dtype=int))
    op_in_dims = tuple(np.arange(0, operator.ndim // 2, dtype=int))
    new_state_dims = tuple(range(len(state_dims)))

    # Apply operator to density matrix: ρ' = O ρ O†
    out_state = state.array
    support_perm = state_dims + tuple(set(tuple(range(out_state.ndim // 2))) - set(state_dims))

    out_state = permute_basis(out_state, support_perm, False)
    out_state = jnp.tensordot(a=operator, b=out_state, axes=(op_out_dims, new_state_dims))

    out_state = _dagger(out_state)
    out_state = jnp.tensordot(a=operator, b=out_state, axes=(op_out_dims, op_in_dims))
    out_state = _dagger(out_state)

    out_state = permute_basis(out_state, support_perm, True)
    return DensityMatrix(out_state)


def apply_kraus_operator(
    kraus: Array,
    array: Array,
    target: tuple[int, ...],
) -> Array:
    """Apply K \\rho K^\\dagger.

    Args:
        kraus (Array): Kraus operator K.
        state (Array): Input density matrix.
        target (tuple[int, ...]): Target qubits.

    Returns:
        Array: K \\rho K^\\dagger.
    """
    state_dims: tuple[int, ...] = target
    n_qubits = int(np.log2(kraus.size))
    kraus = kraus.reshape((2,) * n_qubits)
    op_dims = tuple(np.arange(kraus.ndim // 2, kraus.ndim, dtype=int))

    array = jnp.tensordot(a=kraus, b=array, axes=(op_dims, state_dims))
    new_state_dims = tuple(range(len(state_dims)))
    array = jnp.moveaxis(a=array, source=new_state_dims, destination=state_dims)

    array = jnp.tensordot(a=kraus, b=_dagger(array), axes=(op_dims, state_dims))
    array = _dagger(array)

    return array


def apply_kraus_sum(
    kraus_ops: Array,
    array: Array,
    target: tuple[int, ...],
) -> DensityMatrix:
    """Apply the following evolution as a sum of Kraus operators:
        .. math::
            S(\\rho) = \\sum_i K_i \\rho K_i^\\dagger

    Args:
        kraus_ops (Array): Stacked K_i.
        state (Array): Input array.
        target (tuple[int, ...]): Qubits the operator is defined on.

    Returns:
        DensityMatrix: Output density matrix.
    """

    apply_one_kraus = jax.vmap(
        partial(
            apply_kraus_operator,
            array=array,
            target=target,
        )
    )
    kraus_evol = apply_one_kraus(kraus_ops)
    output_dm = jnp.sum(kraus_evol, 0)
    return DensityMatrix(output_dm)


def filter_noise(noise: NoiseProtocol) -> NoiseProtocol:
    """Return None when all numbers in `error_probability` equal zero.

    Args:
        noise (NoiseProtocol): Noise instance.

    Returns:
        NoiseProtocol: Filtered noise from instances
            when all numbers in`error_probability` equal zero.
    """
    if noise is None:
        return noise

    def check_zero_proba(digital_noise: DigitalNoiseInstance) -> bool:
        if isinstance(digital_noise.error_probability, float):
            return digital_noise.error_probability != 0
        return all(p != 0 for p in digital_noise.error_probability)

    nonzero_noise = tuple(filter(lambda digital_noise: check_zero_proba(digital_noise), noise))
    if not nonzero_noise:
        return None
    return nonzero_noise


def apply_operator_with_noise(
    state: DensityMatrix,
    operator: Array,
    target: tuple[int, ...],
    control: tuple[Union[int, None], ...],
    noise: NoiseProtocol,
) -> State:
    """Evolves the input state and applies a noisy quantum channel
       on the evolved state :math:`\rho`.

        The evolution is represented as a sum of Kraus operators:
        .. math::
            S(\\rho) = \\sum_i K_i \\rho K_i^\\dagger,

    Args:
        state (State): Input state or density matrix.
        operator (Array): Operator to apply.
        target (tuple[int, ...]): Target qubits.
        control (tuple[int  |  None, ...]): Control qubits.
        noise (NoiseProtocol): The noise protocol.

    Returns:
        Array: Output state or density matrix.
    """
    state_gate = apply_operator(state, operator, target, control)
    noise = filter_noise(noise)
    if noise is None:
        return state_gate
    else:
        kraus_ops = jnp.stack(tuple(reduce(add, tuple(n.kraus for n in noise))))
        output_dm = apply_kraus_sum(kraus_ops, state_gate.array, target)
        return output_dm


def group_by_index(gates: Iterable[Primitive]) -> Iterable[Primitive]:
    """Group gates together which are acting on the same qubit."""
    sorted_gates = []
    gate_batch = []
    for gate in gates:
        if not is_controlled(gate.control):
            gate_batch.append(gate)
        else:
            if len(gate_batch) > 0:
                gate_batch.sort(key=lambda g: g.target)
                sorted_gates += gate_batch
                gate_batch = []
            sorted_gates.append(gate)
    if len(gate_batch) > 0:
        gate_batch.sort(key=lambda g: g.target)
        sorted_gates += gate_batch
    return sorted_gates


def merge_operators(
    operators: tuple[Array, ...], targets: tuple[int, ...], controls: tuple[int, ...]
) -> tuple[tuple[Array, ...], tuple[int, ...], tuple[int, ...]]:
    """
    If possible, merge several operators acting on the same qubits into a single array
    which can then be contracted over a state in a single tensordot operation.

    Arguments:
        operators: The arrays representing the unitaries to be merged.
        targets: The corresponding target qubits.
        controls: The corresponding control qubits.

    Returns:
        A tuple of merged operators, targets and controls.
    """
    if len(operators) < 2:
        return operators, targets, controls
    operators, targets, controls = operators[::-1], targets[::-1], controls[::-1]
    merged_operator, merged_target, merged_control = operators[0], targets[0], controls[0]
    merged_operators = merged_targets = merged_controls = tuple()  # type: ignore[var-annotated]
    for operator, target, control in zip(operators[1:], targets[1:], controls[1:]):
        if target == merged_target and control == merged_control:
            merged_operator = merged_operator @ operator
        else:
            merged_operators += (merged_operator,)
            merged_targets += (merged_target,)
            merged_controls += (merged_control,)
            merged_operator, merged_target, merged_control = operator, target, control
    if merged_operator is not None:
        merged_operators += (merged_operator,)
        merged_targets += (merged_target,)
        merged_controls += (merged_control,)
    return merged_operators[::-1], merged_targets[::-1], merged_controls[::-1]


@singledispatch
def apply_gates(
    state: Any,
    gate: Union[Primitive, Iterable[Primitive]],
    values: dict[str, float] = dict(),
    op_type: OperationType = OperationType.UNITARY,
    group_gates: bool = False,  # Defaulting to False since this can be performed once before circuit execution
    merge_ops: bool = True,
) -> Any:
    raise NotImplementedError("apply_gate is not implemented")


def prepare_sequence_reduce(
    gate: Union[Primitive, Iterable[Primitive]],
    values: dict[str, float] = dict(),
    op_type: OperationType = OperationType.UNITARY,
    group_gates: bool = False,  # Defaulting to False since this can be performed once before circuit execution
    merge_ops: bool = True,
) -> tuple[tuple[Array, ...], tuple, tuple, list[NoiseProtocol]]:
    """Prepare the tuples to be used when applying operations.

    Args:
        gate (Union[Primitive, Iterable[Primitive]]): Gate(s) to apply.
        values (dict[str, float], optional): A dictionary with parameter values.
            Defaults to dict().
        op_type (OperationType, optional): The type of operation to perform: Unitary, Dagger or Jacobian.
            Defaults to OperationType.UNITARY.
        group_gates (bool, optional): Group gates together which are acting on the same qubit.
            Defaults to False.

    Returns:
        tuple[tuple[Array, ...], tuple, tuple, list[NoiseProtocol]]: Operators, targets,
            controls and noise.
    """
    operator: tuple[Array, ...]
    noise = list()
    if isinstance(gate, Primitive):
        operator_fn = getattr(gate, op_type)
        operator, target, control = (operator_fn(values),), gate.target, gate.control
        noise += [gate.noise]
    else:
        if group_gates:
            gate = group_by_index(gate)
        operator = tuple(getattr(g, op_type)(values) for g in gate)
        target = reduce(add, [g.target for g in gate])
        control = reduce(add, [g.control for g in gate])
        noise = [g.noise for g in gate]
        # merge when no noise is present
        if (noise == [None] * len(noise)) and merge_ops:
            operator, target, control = merge_operators(operator, target, control)

    return operator, target, control, noise


@apply_gates.register
def _(
    state: Array,
    gate: Union[Primitive, Iterable[Primitive]],
    values: dict[str, float] = dict(),
    op_type: OperationType = OperationType.UNITARY,
    group_gates: bool = False,  # Defaulting to False since this can be performed once before circuit execution
    merge_ops: bool = True,
) -> State:
    """Wrapper function for 'apply_operator' which applies a gate or a series of gates to a given state.
    Arguments:
        state: Array or DensityMatrix to operate on.
        gate: Gate(s) to apply.
        values: A dictionary with parameter values.
        op_type: The type of operation to perform: Unitary, Dagger or Jacobian.
        group_gates: Group gates together which are acting on the same qubit.
        merge_ops: Attempt to merge operators acting on the same qubit.

    Returns:
        Array or density matrix after applying 'gate'.
    """
    operator, target, control, noise = prepare_sequence_reduce(
        gate, values, op_type, group_gates, merge_ops
    )

    # faster way to check has_noise
    has_noise = noise != [None] * len(noise)
    if has_noise:
        state = density_mat(state)
        output_state = reduce(
            lambda state, gate: apply_operator_with_noise(state, *gate),
            zip(operator, target, control, noise),
            state,
        )
    else:
        output_state = reduce(
            lambda state, gate: apply_operator(state, *gate),
            zip(operator, target, control),
            state,
        )
    return output_state


@apply_gates.register
def _(
    state: BCOO,
    gate: Union[Primitive, Iterable[Primitive]],
    values: dict[str, float] = dict(),
    op_type: OperationType = OperationType.UNITARY,
    group_gates: bool = False,  # Defaulting to False since this can be performed once before circuit execution
    merge_ops: bool = True,
) -> State:
    """Wrapper function for 'apply_operator' which applies a gate or a series of gates to a given state.
    Arguments:
        state: Array or DensityMatrix to operate on.
        gate: Gate(s) to apply.
        values: A dictionary with parameter values.
        op_type: The type of operation to perform: Unitary, Dagger or Jacobian.
        group_gates: Group gates together which are acting on the same qubit.
        merge_ops: Attempt to merge operators acting on the same qubit.

    Returns:
        Array or density matrix after applying 'gate'.
    """
    operator, target, control, noise = prepare_sequence_reduce(
        gate, values, op_type, group_gates, merge_ops
    )

    # faster way to check has_noise
    has_noise = noise != [None] * len(noise)
    if has_noise:
        raise NotImplementedError("Noisy simulations are not supported with sparse operators.")
    else:
        output_state = reduce(
            lambda state, gate: apply_operator(state, *gate),
            zip(operator, target, control),
            state,
        )
    return output_state


@apply_gates.register
def _(
    state: DensityMatrix,
    gate: Union[Primitive, Iterable[Primitive]],
    values: dict[str, float] = dict(),
    op_type: OperationType = OperationType.UNITARY,
    group_gates: bool = False,  # Defaulting to False since this can be performed once before circuit execution
    merge_ops: bool = True,
) -> DensityMatrix:
    """Wrapper function for 'apply_operator' which applies a gate or a series of gates to a given state.
    Arguments:
        state: Array or DensityMatrix to operate on.
        gate: Gate(s) to apply.
        values: A dictionary with parameter values.
        op_type: The type of operation to perform: Unitary, Dagger or Jacobian.
        group_gates: Group gates together which are acting on the same qubit.
        merge_ops: Attempt to merge operators acting on the same qubit.

    Returns:
        Array or density matrix after applying 'gate'.
    """
    operator, target, control, noise = prepare_sequence_reduce(
        gate, values, op_type, group_gates, merge_ops
    )
    output_state = reduce(
        lambda state, gate: apply_operator_with_noise(state, *gate),
        zip(operator, target, control, noise),
        state,
    )
    return output_state
