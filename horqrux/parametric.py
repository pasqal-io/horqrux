from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from .matrices import OPERATIONS_DICT
from .primitive import Primitive
from .utils import (
    ControlQubits,
    QubitSupport,
    TargetQubits,
    _jacobian,
    _unitary,
    is_controlled,
)


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
            self.generator_name,
            self.target[0],
            self.control[0],
            self.param,
        )
        return (children, aux_data)

    def __iter__(self) -> Iterable:
        return iter((self.generator_name, self.target, self.control, self.param))

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

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


def RX(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RX gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("X", target, control, param)


def RY(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RY gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("Y", target, control, param)


def RZ(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RZ gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("Z", target, control, param)


class _PHASE(Parametric):
    def unitary(self, values: dict[str, float] = dict()) -> Array:
        u = jnp.eye(2, 2, dtype=jnp.complex128)
        u = u.at[(1, 1)].set(jnp.exp(1.0j * self.parse_values(values)))
        return u

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        jac = jnp.zeros((2, 2), dtype=jnp.complex128)
        jac = jac.at[(1, 1)].set(1j * jnp.exp(1.0j * self.parse_values(values)))
        return jac

    @property
    def name(self) -> str:
        base_name = "PHASE"
        return "C" + base_name if is_controlled(self.control) else base_name


def PHASE(param: float, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """Phase gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """

    return _PHASE("I", target, control, param)
