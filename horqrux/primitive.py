from __future__ import annotations

from .abstract import Operator, QubitSupport
from .matrices import _H, _I, _ISQSWAP, _ISWAP, _NOT, _S, _SQSWAP, _SWAP, _T, _X, _Y, _Z

ControlIdx = QubitSupport
TargetIdx = QubitSupport

# Single qubit gates


def I(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """The Identity operation on a single qubit.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_I, target_idx, control_idx)


def NOT(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """NOT gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: `NOT(((1, ), ), (None, ))` applies the NOT to qubit 1.

    Example usage controlled: `NOT(((1, )), ((0, )))` applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_NOT, target_idx, control_idx)


def X(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """X gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: X(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: X(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_X, target_idx, control_idx)


def Y(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """Y gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Y(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Y(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_Y, target_idx, control_idx)


def Z(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """Z gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Z(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Z(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_Z, target_idx, control_idx)


def H(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """H gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: H(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: H(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones
        describing the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_H, target_idx, control_idx)


def S(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """S gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Y(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Y(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_S, target_idx, control_idx)


def T(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """T gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Y(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Y(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_T, target_idx, control_idx)


## Multi qubit gates


def SWAP(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """SWAP gate. By providing a control idx it turns into a controlled gate (Fredkin gate),
       use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_SWAP, target_idx, control_idx)


def SQSWAP(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """SQSWAP gate. By providing a control idx it turns into a controlled gate (Fredkin gate),
       use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_SQSWAP, target_idx, control_idx)


def ISWAP(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """ISWAP gate. By providing a control idx it turns into a controlled gate
      (Fredkin gate), use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_ISWAP, target_idx, control_idx)


def ISQSWAP(target_idx: TargetIdx, control_idx: ControlIdx = (None,)) -> Operator:
    """ISQSWAP gate. By providing a control idx it turns into a controlled gate
       (Fredkin gate), use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Operator(_ISQSWAP, target_idx, control_idx)
