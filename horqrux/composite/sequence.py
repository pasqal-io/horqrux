from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Iterable

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.utils.operator_utils import State


@register_pytree_node_class
@dataclass
class OpSequence:
    operations: list = field(default_factory=list)

    def tree_flatten(self) -> tuple:
        children = (self.operations,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    @property
    def qubit_support(self) -> tuple:
        return tuple(set().union(*(op.qubit_support for op in self.operations)))

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> State:
        return reduce(lambda state, gate: gate.__call__(state, values), self.operations, state)

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
        return reduce(jnp.matmul, map(lambda x: x.tensor(values, full_support), self.operations))

    def __iter__(self) -> Iterable:
        def flatten(item: Any) -> Iterable:
            # If the item is a OpSequence, iterate through its operations
            if isinstance(item, OpSequence):
                for sub_item in item.operations:
                    yield from flatten(sub_item)
            # If the item is not a OpSequence, yield it directly
            else:
                yield item

        # Iterate through the operations and flatten
        for operation in self.operations:
            yield from flatten(operation)
