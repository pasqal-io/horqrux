from __future__ import annotations

from .abstract import Operator, QubitSupport
from .matrices import _X, _Y, _Z, _unitary

ControlIdx = QubitSupport
TargetIdx = QubitSupport


def Rx(theta: float, target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """Rx gate.

    Args:
        theta (float): Rotational angle.
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    unitary = _unitary(theta) * _X
    return Operator(unitary, target_idx, control_idx)


def Ry(theta: float, target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """Ry gate.

    Args:
        theta (float): Rotational angle.
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    unitary = _unitary(theta) * _Y
    return Operator(unitary, target_idx, control_idx)


def Rz(theta: float, target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """Rz gate.

    Args:
        theta (float): Rotational angle.
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    unitary = _unitary(theta) * _Z
    return Operator(unitary, target_idx, control_idx)
