from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.apply import apply_gates
from horqrux.primitives import Primitive
from horqrux.utils import State, zero_state


@register_pytree_node_class
@dataclass
class OpSequence:
    operations: list[Primitive] = field(default_factory=list)

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
        if state is None:
            state = zero_state(len(self.qubit_support))
        return apply_gates(state, self.operations, values)

    def tensor(self, values: dict[str, float] = dict()) -> Array:
        """Obtain the unitary.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().

        Returns:
            Array: Unitary representation.
        """
        return reduce(jnp.matmul, map(lambda x: x.tensor(values), self.operations))
