from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.primitives import Primitive
from horqrux.utils.operator_utils import State

from .sequence import OpSequence


def operations_to_list(operations: Primitive | OpSequence | list) -> list:
    if not isinstance(operations, list):
        op_list = [operations] if isinstance(operations, Primitive) else operations.operations
    else:
        op_list = operations
    return op_list


@register_pytree_node_class
class Scale(OpSequence):
    """
    Generic container for multiplying a 'Primitive', 'Sequence' or 'Add' instance by a parameter.

    Attributes:
        operations: Operations as a Sequence, Add, or a single Primitive operation.
        parameter_name: Name of the parameter to multiply operations with.
    """

    def __init__(
        self, operations: Primitive | OpSequence | list, parameter_name: str | float
    ) -> None:
        super().__init__(operations_to_list(operations))
        self.parameter: str | float = parameter_name

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> Array:
        param = values[self.parameter] if isinstance(self.parameter, str) else self.parameter
        return jnp.array(param) * super().__call__(state, values)

    def tree_flatten(self) -> tuple:
        children = (self.operations,)
        aux_data = (self.parameter,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def tensor(self, values: dict[str, float] = dict()) -> Array:
        """Obtain the unitary.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().

        Returns:
            Array: Unitary representation.
        """
        param = values[self.parameter] if isinstance(self.parameter, str) else self.parameter
        return param * super().tensor(values)


@register_pytree_node_class
class Add(OpSequence):
    """
    The 'add' operation applies all 'operations' to 'state' and returns the sum of states.

    Attributes:
        operations: List of operations to add up.
    """

    def __init__(self, operations: Primitive | OpSequence | list) -> None:
        super().__init__(operations_to_list(operations))

    def tree_flatten(self) -> tuple:
        children = (self.operations,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> State:
        return reduce(add, map(lambda op: op(state, values), self.operations))

    def tensor(self, values: dict[str, float] = dict()) -> Array:
        """Obtain the unitary.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().

        Returns:
            Array: Unitary representation.
        """
        return reduce(add, map(lambda op: op.tensor(values), self.operations))


@register_pytree_node_class
class Observable(Add):
    """
    The Observable :math:`O` represents an operator from which
    we can extract expectation values from quantum states.

    Given an input state :math:`\\ket\\rangle`, the expectation value with :math:`O` is defined as
    :math:`\\langle\\bra|O\\ket\\rangle`

    Attributes:
        operations: List of operations.
    """

    def __init__(self, operations: Primitive | OpSequence | list) -> None:
        super().__init__(operations_to_list(operations))
