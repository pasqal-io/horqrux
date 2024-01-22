from __future__ import annotations

from .abstract import Primitive
from .utils import ControlQubits, TargetQubits

# Single qubit gates


def I(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """The Identity operation on a single qubit.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("I", target, control)


def NOT(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """NOT gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: `NOT(((1, ), ), (None, ))` applies the NOT to qubit 1.

    Example usage controlled: `NOT(((1, )), ((0, )))` applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("NOT", target, control)


def X(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """X gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: X(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: X(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("X", target, control)


def Y(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """Y gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Y(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Y(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("Y", target, control)


def Z(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """Z gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Z(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Z(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("Z", target, control)


def H(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """H gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: H(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: H(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones
        describing the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("H", target, control)


def S(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """S gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Y(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Y(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("S", target, control)


def T(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """T gate. Note that since we lazily evaluate the circuit, this function
    returns the gate representationt of Gate type and does *not* apply the gate.
    By providing a control idx it turns into a controlled gate, use None for no control qubits.

    Example usage: Y(((1, ), ), (None, )) applies the NOT to qubit 1.
    Example usage controlled: Y(((1, )), ((0, ))) applies the NOT to qubit 1 with controlled bit 0.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("T", target, control)


## Multi qubit gates


def SWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """SWAP gate. By providing a control idx it turns into a controlled gate (Fredkin gate),
       use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("SWAP", target, control)


def SQSWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """SQSWAP gate. By providing a control idx it turns into a controlled gate (Fredkin gate),
       use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("SQSWAP", target, control)


def ISWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """ISWAP gate. By providing a control idx it turns into a controlled gate
      (Fredkin gate), use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("ISWAP", target, control)


def ISQSWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """ISQSWAP gate. By providing a control idx it turns into a controlled gate
       (Fredkin gate), use None for no control qubits.

    Example usage: SWAP(((0, 1), ), (None, )) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Primitive("ISQSWAP", target, control)
