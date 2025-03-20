from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Any

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

    def __call__(self, state: State, values: dict[str, Array] = dict()) -> State:
        return reduce(lambda state, gate: gate.__call__(state, values), self.operations, state)

    def tensor(self, values: dict[str, float] = dict()) -> Array:
        """Obtain the unitary.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().

        Returns:
            Array: Unitary representation.
        """
        return reduce(jnp.matmul, map(lambda x: x.tensor(values), self.operations))
