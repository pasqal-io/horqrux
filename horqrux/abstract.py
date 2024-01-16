from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from .utils import QubitSupport


@register_pytree_node_class
@dataclass
class Operator:
    unitary: ArrayLike
    target: QubitSupport
    control: QubitSupport

    def __post_init__(self) -> None:
        def _parse(idx: QubitSupport | Tuple[None, ...]) -> QubitSupport:
            return (idx,) if isinstance(idx, int) or idx is None else idx

        self.target, self.control = list(map(_parse, (self.target, self.control)))

    def __iter__(self) -> Iterable:
        return iter((self.unitary, self.target, self.control))

    def tree_flatten(self) -> Tuple[Tuple[Array], Tuple[QubitSupport, QubitSupport]]:
        children = (self.unitary,)
        aux_data = (
            self.target,
            self.control,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)
