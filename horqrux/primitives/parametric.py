from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO, bcoo_concatenate, empty
from jax.tree_util import register_pytree_node_class

from horqrux._misc import default_complex_dtype
from horqrux.noise import NoiseProtocol
from horqrux.utils.operator_utils import (
    ControlQubits,
    QubitSupport,
    TargetQubits,
    _jacobian,
    _unitary,
    is_controlled,
)
from horqrux.utils.sparse_utils import eigvalsh_sp

from .primitive import Primitive

default_dtype = default_complex_dtype()
nonzero_jit = jax.jit(jnp.nonzero, static_argnames="size")
unique_jit = jax.jit(jnp.unique, static_argnames="size")


@register_pytree_node_class
@dataclass
class Parametric(Primitive):
    """Extension of the Primitive class adding the option to pass a parameter."""

    generator_name: str
    target: QubitSupport
    control: QubitSupport
    noise: NoiseProtocol = None
    sparse: bool = False
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
    ) -> tuple[tuple, tuple[str, tuple, tuple, NoiseProtocol, bool, str | float, float]]:
        children = ()
        aux_data = (
            self.generator_name,
            self.target[0],
            self.control[0],
            self.noise,
            self.sparse,
            self.param,
            self.shift,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def _unitary(self, values: dict[str, float] = dict()) -> Array:
        return _unitary(self.generator, self.parse_values(values))

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        return _jacobian(self.generator, self.parse_values(values))

    @property
    def name(self) -> str:
        base_name = "R" + self.generator_name
        return "C" + base_name if is_controlled(self.control) else base_name

    def __repr__(self) -> str:
        return (
            self.name
            + f"(target={self.target}, control={self.control}, param={self.param}, shift={self.shift})"
        )

    @cached_property
    def eigenvals_generator(self) -> Array:
        """Get eigenvalues of the underlying operation.

        Arguments:
            values: Parameter values.

        Returns:
            Array: Eigenvalues of the operation.
        """
        eig_vals_generator = eigvalsh_sp(self.generator)
        if is_controlled(self.control):
            if not self.sparse:
                eig_vals_generator = jnp.concatenate(
                    (
                        jnp.zeros(2 ** (len(self.control)), dtype=eig_vals_generator.dtype),
                        eig_vals_generator,
                    )
                )
            else:
                eig_vals_generator = bcoo_concatenate(
                    empty(2 ** (len(self.control)), dtype=eig_vals_generator.dtype),
                    eig_vals_generator,
                )
        return eig_vals_generator

    @cached_property
    def spectral_gap(self) -> Array:
        """Difference between the moduli of the two largest eigenvalues of the generator.

        Returns:
            Array: Spectral gap value.
        """
        spectrum = jnp.atleast_2d(self.eigenvals_generator)
        diffs = spectrum - spectrum.T
        # note for jitting, must specify a size
        # atm only size 2 is acceptable given all possible generators in OPERATIONS_DICT
        spectral_gap = unique_jit(jnp.abs(jnp.tril(diffs)), size=2)
        return spectral_gap[nonzero_jit(spectral_gap, size=1)]

    @property
    def is_parametric(self) -> bool:
        return isinstance(self.param, str)


def RX(
    param: float | str,
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
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
    return Parametric("X", target, control, noise, param=param, sparse=sparse)


def RY(
    param: float | str,
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
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
    return Parametric("Y", target, control, noise, param=param, sparse=sparse)


def RZ(
    param: float | str,
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
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
    return Parametric("Z", target, control, noise, param=param, sparse=sparse)


class _PHASE(Parametric):
    def _unitary(self, values: dict[str, float] = dict()) -> Array:
        u = jnp.eye(2, 2, dtype=default_dtype)
        u = u.at[(1, 1)].set(jnp.exp(1.0j * self.parse_values(values)))
        if self.sparse:
            u = BCOO.fromdense(u)
        return u

    def jacobian(self, values: dict[str, float] = dict()) -> Array:
        jac = jnp.zeros((2, 2), dtype=default_dtype)
        jac = jac.at[(1, 1)].set(1j * jnp.exp(1.0j * self.parse_values(values)))
        if self.sparse:
            jac = BCOO.fromdense(jac)
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
    sparse: bool = False,
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

    return _PHASE("I", target, control, noise, param=param, sparse=sparse)
