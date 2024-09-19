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
    param_name: str | None = None
    param_val: float = 0.0
    shift: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        def parse_dict(self: Parametric, values: dict[str, float] = dict()) -> float:
            return values[self.param_name] + self.shift  # type: ignore[index]

        def parse_val(self: Parametric, values: dict[str, float] = dict()) -> float:
            return self.param_val + self.shift  # type: ignore[return-value]

        self.parse_values = parse_val if self.param_name is None else parse_dict

    def tree_flatten(  # type: ignore[override]
        self,
    ) -> Tuple[Tuple[float, float], Tuple[str, Tuple, Tuple, str | None]]:
        children = (self.param_val, self.shift)
        aux_data = (self.generator_name, self.target[0], self.control[0], self.param_name)
        return (children, aux_data)

    def __iter__(self) -> Iterable:
        return iter(
            (
                self.generator_name,
                self.target,
                self.control,
                self.param_name,
                self.param_val,
                self.shift,
            )
        )

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*aux_data, *children)

    def unitary(self, values: dict[str, float] = dict()) -> Array:
        return _unitary(OPERATIONS_DICT[self.generator_name], self.parse_values(self, values))

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        return _jacobian(OPERATIONS_DICT[self.generator_name], self.parse_values(self, values))

    @property
    def name(self) -> str:
        base_name = "R" + self.generator_name
        return "C" + base_name if is_controlled(self.control) else base_name

    def __repr__(self) -> str:
        return (
            self.name
            + f"(target={self.target[0]},"
            + f"control={self.control[0]},"
            + f"param_name={self.param_name},"
            + f"param_val={self.param_val},"
            + f"shift={self.shift})"
        )

    def set_shift(self, shift: float) -> None:
        self.shift = shift


def RX(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RX gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """

    if isinstance(param, str):
        return Parametric("X", target, control, param_name=param)
    return Parametric("X", target, control, param_val=param)


def RY(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RY gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    if isinstance(param, str):
        return Parametric("Y", target, control, param_name=param)
    return Parametric("Y", target, control, param_val=param)


def RZ(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RZ gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    if isinstance(param, str):
        return Parametric("Z", target, control, param_name=param)
    return Parametric("Z", target, control, param_val=param)


class _PHASE(Parametric):
    def unitary(self, values: dict[str, float] = dict()) -> Array:
        u = jnp.eye(2, 2, dtype=jnp.complex128)
        u = u.at[(1, 1)].set(jnp.exp(1.0j * self.parse_values(self, values)))
        return u

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        jac = jnp.zeros((2, 2), dtype=jnp.complex128)
        jac = jac.at[(1, 1)].set(1j * jnp.exp(1.0j * self.parse_values(self, values)))
        return jac

    @property
    def name(self) -> str:
        base_name = "PHASE"
        return "C" + base_name if is_controlled(self.control) else base_name


def PHASE(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """Phase gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    if isinstance(param, str):
        return _PHASE("I", target, control, param_name=param)
    return _PHASE("I", target, control, param_val=param)
