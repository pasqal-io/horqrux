from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.apply import apply_gate
from horqrux.primitives import Primitive
from horqrux.utils import State, add, zero_state

from .sequence import Sequence


@register_pytree_node_class
@dataclass
class Scale(Sequence):
    def __init__(self, operations: Primitive | Sequence, parameter_name: str) -> None:
        op_list = [operations] if isinstance(operations, Primitive) else operations.operations
        super().__init__(op_list)
        self.parameter: str = parameter_name

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> Array:
        return jnp.array(values[self.parameter]) * super().__call__(state, values)

    def tree_flatten(self) -> tuple:
        children = (self.operations,)
        aux_data = (self.parameter,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)


@register_pytree_node_class
@dataclass
class Add(Sequence):
    def __init__(self, operations: Primitive | Sequence) -> None:
        op_list = [operations] if isinstance(operations, Primitive) else operations.operations
        super().__init__(op_list)

    def tree_flatten(self) -> tuple:
        children = (self.operations,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> State:
        if state is None:
            state = zero_state(len(self.qubit_support))
        return reduce(add, (apply_gate(state, op, values) for op in self.operations))
