from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

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
    """Abstract class which stores information about generators target and control qubits
    of a particular quantum operator."""

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

    def tree_flatten(self) -> Tuple[Tuple, Tuple[str, TargetQubits, ControlQubits]]:
        children = ()
        aux_data = (self.generator_name, self.target, self.control)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def unitary(self, values: dict[str, float] = dict()) -> Array:
        return OPERATIONS_DICT[self.generator_name]

    def dagger(self, values: dict[str, float] = dict()) -> Array:
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
    """Extension of the Primitive class adding the option to pass a parameter."""

    generator_name: str
    target: QubitSupport
    control: QubitSupport
    param: str | float = ""

    def __post_init__(self) -> None:
        super().__post_init__()

        def parse_dict(values: dict[str, float] = dict()) -> float:
            return values[self.param]  # type: ignore[index]

        def parse_val(values: dict[str, float] = dict()) -> float:
            return self.param  # type: ignore[return-value]

        self.parse_values = parse_dict if isinstance(self.param, str) else parse_val

    def tree_flatten(self) -> Tuple[Tuple, Tuple[str, Tuple, Tuple, str | float]]:  # type: ignore[override]
        children = ()
        aux_data = (
            self.name,
            self.target,
            self.control,
            self.param,
        )
        return (children, aux_data)

    def unitary(self, values: dict[str, float] = dict()) -> Array:
        return _unitary(OPERATIONS_DICT[self.generator_name], self.parse_values(values))

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        return _jacobian(OPERATIONS_DICT[self.generator_name], self.parse_values(values))

    @property
    def name(self) -> str:
        base_name = "R" + self.generator_name
        return "C" + base_name if is_controlled(self.control) else base_name

    def __repr__(self) -> str:
        return (
            self.name + f"(target={self.target[0]}, control={self.control[0]}, param={self.param})"
        )
