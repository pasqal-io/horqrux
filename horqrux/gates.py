from __future__ import annotations

import jax.numpy as jnp

from .types import ControlIdx, Gate, TargetIdx

# Single qubit gates
# NOT gate
_NOT = jnp.asarray([[0, 1], [1, 0]])


def NOT(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_NOT, target_idx, control_idx)  # type: ignore[arg-type]


# X gate
_X = jnp.array([[0, 1], [1, 0]])


def X(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_X, target_idx, control_idx)  # type: ignore[arg-type]


# Y gate
_Y = jnp.array(
    [[0, -1j], [1j, 0]],
)


def Y(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_Y, target_idx, control_idx)  # type: ignore[arg-type]


# Z gate
_Z = jnp.array(
    [[1, 0], [0, -1]],
)


def Z(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_Z, target_idx, control_idx)  # type: ignore[arg-type]


# H gate
_H = (
    jnp.array(
        [[1, 1], [1, -1]],
    )
    * 1
    / jnp.sqrt(2)
)


def H(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_H, target_idx, control_idx)  # type: ignore[arg-type]


# S gate
_S = jnp.array(
    [[1, 0], [0, 1j]],
)


def S(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_S, target_idx, control_idx)  # type: ignore[arg-type]


# T gate
_T = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]])


def T(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_T, target_idx, control_idx)  # type: ignore[arg-type]


# Rotation gates
def Rx(theta: float, target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
    """Rx gate.

    Args:
        theta (float): Rotational angle.
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    O = jnp.cos(theta / 2) * jnp.eye(2) - 1j * jnp.sin(theta / 2) * _X
    return Gate(O, target_idx, control_idx)  # type: ignore[arg-type]


def Ry(theta: float, target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
    """Ry gate.

    Args:
        theta (float): Rotational angle.
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    O = jnp.cos(theta / 2) * jnp.eye(2) - 1j * jnp.sin(theta / 2) * _Y
    return Gate(O, target_idx, control_idx)  # type: ignore[arg-type]


def Rz(theta: float, target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
    """Rz gate.

    Args:
        theta (float): Rotational angle.
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    O = jnp.cos(theta / 2) * jnp.eye(2) - 1j * jnp.sin(theta / 2) * _Z
    return Gate(O, target_idx, control_idx)  # type: ignore[arg-type]


## Multi qubit gates


_SWAP = jnp.asarray(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
)


def SWAP(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_SWAP, target_idx, control_idx)  # type: ignore[arg-type]


_SQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
        [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
        [0, 0, 0, 1],
    ]
)


def SQSWAP(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_SQSWAP, target_idx, control_idx)  # type: ignore[arg-type]


_ISWAP = jnp.asarray(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
)


def ISWAP(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_ISWAP, target_idx, control_idx)  # type: ignore[arg-type]


_ISQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 1 / jnp.sqrt(2), 1j / jnp.sqrt(2), 0],
        [0, 1j / jnp.sqrt(2), 1 / jnp.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
)


def ISQSWAP(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
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
    return Gate(_ISQSWAP, target_idx, control_idx)  # type: ignore[arg-type]


_I = jnp.asarray([[1, 0], [0, 1]])


def I(target_idx: TargetIdx, control_idx: ControlIdx = None) -> Gate:
    """The Identity operation on a single qubit.

    Args:
        target_idx (TargetIdx): Tuple of Tuples describing the qubits to apply to.
        control_idx (ControlIdx, optional): Tuple of Tuples or Nones describing
        the control qubits of length(target_idx). Defaults to (None,).

    Returns:
        Gate: Gate object.
    """
    return Gate(_I, target_idx, control_idx)  # type: ignore[arg-type]
