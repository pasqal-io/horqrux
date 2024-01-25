from __future__ import annotations

from .abstract import Primitive
from .utils import ControlQubits, TargetQubits

# Single qubit gates


def I(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """Identity / I gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: I(1) applies I to qubit 1.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("I", target, control)


def X(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """X gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: X(1) applies X to qubit 1.
    Example usage controlled: X(1, 0) applies CX / CNOT to qubit 1 with controlled qubit 0.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("X", target, control)


NOT = X


def Y(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """Y gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: Y(1) applies X to qubit 1.
    Example usage controlled: Y(1, 0) applies CY to qubit 1 with controlled qubit 0.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("Y", target, control)


def Z(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """Z gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: Z(1) applies Z to qubit 1.
    Example usage controlled: Z(1, 0) applies CZ to qubit 1 with controlled qubit 0.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("Z", target, control)


def H(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """H/ Hadamard gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: H(1) applies Hadamard to qubit 1.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("H", target, control)


def S(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """S gate or constant phase gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: S(1) applies S to qubit 1.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("S", target, control)


def T(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """T gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: T(1) applies Hadamard to qubit 1.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("T", target, control)


## Multi (target) qubit gates


def SWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    """SWAP gate. By providing a control, it turns into a controlled gate (Fredkin gate),
       use None for no control qubits.

    Example usage: SWAP((0, 1), (None,)) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target: Tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints or None describing
        the control qubits. Defaults to (None,).

    Returns:
        A Primitive instance.
    """
    return Primitive("SWAP", target, control)


def SQSWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    return Primitive("SQSWAP", target, control)


def ISWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    return Primitive("ISWAP", target, control)


def ISQSWAP(target: TargetQubits, control: ControlQubits = (None,)) -> Primitive:
    return Primitive("ISQSWAP", target, control)
