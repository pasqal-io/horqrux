from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple

import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class

from .matrices import OPERATIONS_DICT
from .utils import (
    ControlQubits,
    QubitSupport,
    TargetQubits,
    _dagger,
    _jacobian,
    _unitary,
    is_controlled,
    none_like,
)


@register_pytree_node_class
@dataclass
class Operator:
    generator_name: str
    target: QubitSupport
    control: QubitSupport

    @staticmethod
    def parse_idx(
        idx: Tuple,
    ) -> Tuple:
        if isinstance(idx, (int, np.int64)):
            return ((idx,),)
        elif isinstance(idx, tuple):
            return (idx,)
        else:
            return (idx.astype(int),)

    def __post_init__(self) -> None:
        self.target = Operator.parse_idx(self.target)
        if self.control is None:
            self.control = none_like(self.target)
        else:
            self.control = Operator.parse_idx(self.control)

    def __iter__(self) -> Iterable:
        return iter((self.generator_name, self.target, self.control))

    def tree_flatten(self) -> Tuple[Tuple, Tuple[str, QubitSupport, QubitSupport]]:
        children = ()
        aux_data = (self.generator_name, self.target, self.control)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def unitary(self, values: dict[str, float] = {}) -> Array:
        return OPERATIONS_DICT[self.generator_name]

    def dagger(self, values: dict[str, float] = {}) -> Array:
        return _dagger(self.unitary(values))

    @property
    def name(self) -> str:
        return "C" + self.generator_name if is_controlled(self.control) else self.generator_name

    def __repr__(self) -> str:
        return self.name + f"(target={self.target[0]}, control={self.control[0]})"


Primitive = Operator


@register_pytree_node_class
@dataclass
class Parametric(Primitive):
    generator_name: str
    target: QubitSupport
    control: QubitSupport
    param: str | float = ""

    def __post_init__(self) -> None:
        super().__post_init__()

        def parse_dict(values: dict[str, float] = {}) -> float:
            return values[self.param]

        self.parse_values: Callable[[dict[str, float]], float] = (
            parse_dict if isinstance(self.param, str) else lambda x: self.param
        )

    def tree_flatten(self) -> Tuple[Tuple, Tuple[str, TargetQubits, ControlQubits, str | float]]:
        children = ()
        aux_data = (
            self.name,
            self.target,
            self.control,
            self.param,
        )
        return (children, aux_data)

    def unitary(self, values: dict[str, float] = {}) -> Array:
        return _unitary(OPERATIONS_DICT[self.generator_name], self.parse_values(values))

    def jacobian(self, values: dict[str, float] = {}) -> Array:
        return _jacobian(OPERATIONS_DICT[self.generator_name], self.parse_values(values))

    @property
    def name(self) -> str:
        base_name = "R" + self.generator_name
        return "C" + base_name if is_controlled(self.control) else base_name

    def __repr__(self) -> str:
        return (
            self.name + f"(target={self.target[0]}, control={self.control[0]}, param={self.param})"
        )
