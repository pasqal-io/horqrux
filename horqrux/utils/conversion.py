from __future__ import annotations

from functools import singledispatch
from typing import Any

from jax import Array
from jax.experimental.sparse import BCOO

from horqrux.circuit import QuantumCircuit
from horqrux.composite.compose import Scale
from horqrux.composite.sequence import OpSequence
from horqrux.primitives.parametric import Parametric
from horqrux.primitives.primitive import Primitive


@singledispatch
def to_sparse(operations: Any) -> Any:
    raise NotImplementedError(
        f"to_sparse is not implemented for this argument type: {type(operations)}"
    )


@singledispatch
def to_dense(operations: Any) -> Any:
    raise NotImplementedError(
        f"to_dense is not implemented for this argument type: {type(operations)}"
    )


@to_sparse.register
def _(operations: Array) -> BCOO:
    return BCOO.fromdense(operations)


@to_dense.register
def _(operations: BCOO) -> Array:
    return operations.todense()


@to_sparse.register
def _(operations: Primitive) -> Any:
    children, aux_data = operations.tree_flatten()
    return operations.tree_unflatten(aux_data[:-1] + (True,), children)


@to_dense.register
def _(operations: Primitive) -> Any:
    children, aux_data = operations.tree_flatten()
    return operations.tree_unflatten(aux_data[:-1] + (False,), children)


@to_sparse.register
def _(operations: Parametric) -> Any:
    children, aux_data = operations.tree_flatten()
    return operations.tree_unflatten(aux_data[:-3] + (True,) + aux_data[-2:], children)


@to_dense.register
def _(operations: Parametric) -> Any:
    children, aux_data = operations.tree_flatten()
    return operations.tree_unflatten(aux_data[:-3] + (False,) + aux_data[-2:], children)


@to_sparse.register
def _(operations: OpSequence) -> OpSequence:
    return type(operations)([to_sparse(op) for op in operations.operations])


@to_dense.register
def _(operations: OpSequence) -> OpSequence:
    return type(operations)([to_dense(op) for op in operations.operations])


@to_sparse.register
def _(operations: Scale) -> Scale:
    return Scale(to_sparse(operations.operations[0]), operations.parameter)


@to_dense.register
def _(operations: Scale) -> Scale:
    return Scale(to_dense(operations.operations[0]), operations.parameter)


@to_sparse.register
def _(operations: QuantumCircuit) -> QuantumCircuit:
    return QuantumCircuit(operations.n_qubits, [to_sparse(op) for op in operations.operations])


@to_dense.register
def _(operations: QuantumCircuit) -> QuantumCircuit:
    return QuantumCircuit(operations.n_qubits, [to_dense(op) for op in operations.operations])
