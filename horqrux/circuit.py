from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.apply import apply_gate
from horqrux.parametric import RX, RY, Parametric
from horqrux.primitive import NOT, Primitive
from horqrux.utils import zero_state


@register_pytree_node_class
@dataclass
class Circuit:
    """A minimalistic circuit class to store a sequence of gates."""

    n_qubits: int
    feature_map: list[Primitive]
    ansatz: list[Primitive]

    def __post_init__(self) -> None:
        self.state = zero_state(self.n_qubits)

    def __call__(self, param_values: Array) -> Array:
        return apply_gate(
            self.state,
            self.feature_map + self.ansatz,
            {name: val for name, val in zip(self.param_names, param_values)},
        )

    @property
    def param_names(self) -> list[str]:
        return [str(op.param_name) for op in self.ansatz if isinstance(op, Parametric)]

    @property
    def n_vparams(self) -> int:
        return len(self.param_names)

    def tree_flatten(self) -> tuple:
        children = (self.feature_map, self.ansatz)
        aux_data = (self.n_qubits,)
        return (aux_data, children)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*aux_data, *children)


def hea(n_qubits: int, n_layers: int, rot_fns: list[Callable] = [RX, RY, RX]) -> list[Primitive]:
    """Hardware-efficient ansatz; A helper function to generate a sequence of rotations followed
    by a global entangling operation."""
    gates = []
    param_names = []
    for _ in range(n_layers):
        for i in range(n_qubits):
            ops = [
                fn(str(uuid4()), qubit)
                for fn, qubit in zip(rot_fns, [i for _ in range(len(rot_fns))])
            ]
            param_names += [op.param_name for op in ops]
            ops += [NOT((i + 1) % n_qubits, i % n_qubits) for i in range(n_qubits)]  # type: ignore[arg-type]
            gates += ops

    return gates
