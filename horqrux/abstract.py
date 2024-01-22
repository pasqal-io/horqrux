from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from jax import Array
from jax.tree_util import register_pytree_node_class

from .matrices import OPERATIONS_DICT
from .utils import QubitSupport, _dagger, _jacobian, _unitary


@register_pytree_node_class
@dataclass
class Operator:
    name: str
    target: QubitSupport
    control: QubitSupport

    def __post_init__(self) -> None:
        def _parse(idx: QubitSupport | Tuple[None, ...]) -> QubitSupport:
            return (idx,) if isinstance(idx, int) or idx is None else idx

        self.target, self.control = list(map(_parse, (self.target, self.control)))

    def __iter__(self) -> Iterable:
        return iter((self.name, self.target, self.control))

    def tree_flatten(self) -> Tuple[Tuple, Tuple[str, QubitSupport, QubitSupport]]:
        children = ()
        aux_data = (self.name, self.target, self.control)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def unitary(self, values: dict[str, float] = {}) -> Array:
        return OPERATIONS_DICT[self.name]

    def dagger(self, values: dict[str, float] = {}) -> Array:
        return _dagger(self.unitary(values))


Primitive = Operator


@register_pytree_node_class
@dataclass
class Parametric(Primitive):
    name: str
    target: QubitSupport
    control: QubitSupport
    param: str | int = ""

    def __post_init__(self) -> None:
        def parse_dict(self, values: dict[str, float] | float = {}) -> float:
            return values[self.param]

        self.parse_values = parse_dict if isinstance(self.param, str) else lambda x: self.param

    def tree_flatten(self) -> Tuple[Tuple, Tuple[str, QubitSupport, QubitSupport, str]]:
        children = ()
        aux_data = (
            self.name,
            self.target,
            self.control,
            self.param,
        )
        return (children, aux_data)

    def unitary(self, values: dict[str, float] = {}) -> Array:
        return _unitary(OPERATIONS_DICT[self.name], self.parse_values(values))

    def jacobian(self, values: dict[str, float] = {}) -> Array:
        return _jacobian(OPERATIONS_DICT[self.name], values[self.param])
