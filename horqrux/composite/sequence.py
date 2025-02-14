from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Any

from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.apply import apply_gate
from horqrux.primitives import Primitive
from horqrux.utils import DensityMatrix, zero_state


@register_pytree_node_class
@dataclass
class Sequence:
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
        all_qubits = reduce(
            lambda x, y: x + y, [op.qubit_support for op in self.operations]
        )  # type:ignore[operator]
        return tuple(set(all_qubits))

    def __call__(
        self, state: Array | DensityMatrix | None = None, values: dict[str, Array] = dict()
    ) -> Array:
        if state is None:
            state = zero_state(len(self.qubit_support))
        return apply_gate(state, self.operations, values)
