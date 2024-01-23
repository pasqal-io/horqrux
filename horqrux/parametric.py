from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .abstract import Parametric
from .utils import ControlQubits, TargetQubits


def RX(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RX gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("X", target, control, param=param)


def RY(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RY gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("Y", target, control, param=param)


def RZ(param: float | str, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """RZ gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """
    return Parametric("Z", target, control, param=param)


def PHASE(param: float, target: TargetQubits, control: ControlQubits = (None,)) -> Parametric:
    """Phase gate.

    Arguments:
        param: Parameter denoting the Rotational angle.
        target: Tuple of target qubits denoted as ints.
        control: Optional tuple of control qubits denoted as ints.

    Returns:
        Parametric: A Parametric gate object.
    """

    def unitary(values: dict[str, float] = {}) -> Array:
        u = jnp.eye(2, 2, dtype=jnp.complex128)
        u = u.at[(1, 1)].set(jnp.exp(1.0j * values[param]))
        return u

    def jacobian(values: dict[str, float] = {}) -> Array:
        jac = jnp.zeros((2, 2), dtype=jnp.complex128)
        jac = jac.at[(1, 1)].set(1j * jnp.exp(1.0j * values[param]))
        return jac

    phase = Parametric("I", target, control, param)
    phase.name = "PHASE"
    phase.unitary = unitary
    phase.jacobian = jacobian
    return phase
