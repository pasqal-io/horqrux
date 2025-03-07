from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux._misc import default_complex_dtype
from horqrux.matrices import OPERATIONS_DICT
from horqrux.noise import NoiseProtocol
from horqrux.utils import (
    ControlQubits,
    QubitSupport,
    TargetQubits,
    _jacobian,
    _unitary,
    is_controlled,
)

from .primitive import Primitive

default_dtype = default_complex_dtype()


@register_pytree_node_class
@dataclass
class Parametric(Primitive):
    """Extension of the Primitive class adding the option to pass a parameter."""

    generator_name: str
    target: QubitSupport
    control: QubitSupport
    noise: NoiseProtocol = None
    param: str | float = ""
    shift: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        def parse_dict(values: dict[str, float] = dict()) -> float:
            # note: shift is for GPSR when the same param_name is used in many operations
            return values[self.param] + self.shift  # type: ignore[index]

        def parse_val(values: dict[str, float] = dict()) -> float:
            return self.param + self.shift  # type: ignore[return-value, operator]

        self.parse_values = parse_dict if isinstance(self.param, str) else parse_val

    def tree_flatten(  # type: ignore[override]
        self,
    ) -> tuple[tuple, tuple[str, tuple, tuple, NoiseProtocol, str | float, float]]:
        children = ()
        aux_data = (
            self.generator_name,
            self.target[0],
            self.control[0],
            self.noise,
            self.param,
            self.shift,
        )
        return (children, aux_data)

    def __iter__(self) -> Iterable:
        return iter(
            (self.generator_name, self.target, self.control, self.noise, self.param, self.shift)
        )

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def _unitary(self, values: dict[str, float] = dict()) -> Array:
        return _unitary(OPERATIONS_DICT[self.generator_name], self.parse_values(values))

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        return _jacobian(OPERATIONS_DICT[self.generator_name], self.parse_values(values))

    @property
    def name(self) -> str:
        base_name = "R" + self.generator_name
        return "C" + base_name if is_controlled(self.control) else base_name

    def __repr__(self) -> str:
        return (
            self.name
            + f"(target={self.target}, control={self.control}, param={self.param}, shift={self.shift})"
        )


def RX(
    param: float | str,
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
) -> Parametric:
    """RX gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.
        noise: The noise instance. Defaults to None.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("X", target, control, noise, param)


def RY(
    param: float | str,
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
) -> Parametric:
    """RY gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.
        noise: The noise instance. Defaults to None.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("Y", target, control, noise, param)


def RZ(
    param: float | str,
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
) -> Parametric:
    """RZ gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.
        noise: The noise instance. Defaults to None.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("Z", target, control, noise, param)


class _PHASE(Parametric):
    def _unitary(self, values: dict[str, float] = dict()) -> Array:
        u = jnp.eye(2, 2, dtype=default_dtype)
        u = u.at[(1, 1)].set(jnp.exp(1.0j * self.parse_values(values)))
        return u

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        jac = jnp.zeros((2, 2), dtype=default_dtype)
        jac = jac.at[(1, 1)].set(1j * jnp.exp(1.0j * self.parse_values(values)))
        return jac

    @property
    def name(self) -> str:
        base_name = "PHASE"
        return "C" + base_name if is_controlled(self.control) else base_name


def PHASE(
    param: float,
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
) -> Parametric:
    """Phase gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.
        noise: The noise instance. Defaults to None.

    Returns:
        Parametric: A Parametric gate object.
    """

    return _PHASE("I", target, control, noise, param)
