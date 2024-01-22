from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .abstract import Parametric
from .utils import ControlQubits, TargetQubits


def Rx(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """Rx gate.

    Args:
        theta (float): Rotational angle.
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Parametric("X", target, control, param=param)


def Ry(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """Ry gate.

    Args:
        theta (float): Rotational angle.
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Parametric("Y", target, control, param=param)


def Rz(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """Rz gate.

    Args:
        theta (float): Rotational angle.
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Parametric("Z", target, control, param=param)


def Phase(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    def unitary(values: dict[str, float] = {}) -> Array:
        u = jnp.eye(2, 2, dtype=jnp.complex128)
        u = u.at[(1, 1)].set(jnp.exp(1.0j * values[param]))
        return u

    def jacobian(values: dict[str, float] = {}) -> Array:
        jac = jnp.zeros((2, 2), dtype=jnp.complex128)
        jac = jac.at[(1, 1)].set(1j * jnp.exp(1.0j * values[param]))
        return jac

    phase = Parametric("I", target, control)
    phase.name = "PHASE"
    phase.unitary = unitary
    phase.jacobian = jacobian
    return phase
