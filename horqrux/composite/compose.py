from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, Iterable

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.primitives import Primitive
from horqrux.utils.operator_utils import State

from .sequence import OpSequence


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
        super().__init__(list(operations))  # type:ignore[arg-type]
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

    def __iter__(self) -> Iterable[Scale]:
        return iter((self,))

    def tensor(
        self,
        values: dict[str, float] = dict(),
        full_support: tuple[int, ...] | None = None,
    ) -> Array:
        """Obtain the tensor representation.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().
            full_support (tuple[int, ...], optional): The qubit support of definition for the unitary.

        Returns:
            Array: Unitary representation.
        """
        param = values[self.parameter] if isinstance(self.parameter, str) else self.parameter
        return param * super().tensor(values, full_support)


@register_pytree_node_class
class Add(OpSequence):
    """
    The 'add' operation applies all 'operations' to 'state' and returns the sum of states.

    Attributes:
        operations: List of operations to add up.
    """

    def __init__(self, operations: Primitive | OpSequence | list) -> None:
        super().__init__(list(operations))  # type:ignore[arg-type]

    def tree_flatten(self) -> tuple:
        children = (self.operations,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> State:
        return reduce(add, map(lambda op: op(state, values), self.operations))

    def tensor(
        self,
        values: dict[str, float] = dict(),
        full_support: tuple[int, ...] | None = None,
    ) -> Array:
        """Obtain the unitary.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().
            full_support (tuple[int, ...], optional): The qubit support of definition for the unitary.

        Returns:
            Array: Unitary representation.
        """
        if full_support is None:
            full_support = self.qubit_support
        elif not set(self.qubit_support).issubset(set(full_support)):
            raise ValueError(
                "Expanding tensor operation requires a `full_support` argument "
                "larger than or equal to the `qubit_support`."
            )
        return reduce(add, map(lambda op: op.tensor(values, full_support), self.operations))


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
        super().__init__(list(operations))  # type:ignore[arg-type]

    def forward(self, state: State | None = None, values: dict[str, Array] = dict()) -> State:
        return super().__call__(state, values)

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> Array:
        """Compute the expectation value using the observable.

        Args:
            state (State | None, optional): Input state. Defaults to None.
            values (dict[str, Array], optional): Parameter values. Defaults to dict().

        Returns:
            Array: Expectation values.
        """
        from horqrux.differentiation.ad import _ad_expectation_single_observable

        return _ad_expectation_single_observable(state, self, values)
