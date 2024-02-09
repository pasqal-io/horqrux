from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy import log2

State = ArrayLike
QubitSupport = Tuple[Any, ...]
ControlQubits = Tuple[Union[None, Tuple[int, ...]], ...]
TargetQubits = Tuple[Tuple[int, ...], ...]
ATOL = 1e-014


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class OperationType(StrEnum):
    UNITARY = "unitary"
    DAGGER = "dagger"
    JACOBIAN = "jacobian"


def _dagger(operator: Array) -> Array:
    return jnp.conjugate(operator.T)


def _unitary(generator: Array, theta: float) -> Array:
    return (
        jnp.cos(theta / 2) * jnp.eye(2, dtype=jnp.complex128) - 1j * jnp.sin(theta / 2) * generator
    )


def _jacobian(generator: Array, theta: float) -> Array:
    return (
        -1
        / 2
        * (jnp.sin(theta / 2) * jnp.eye(2, dtype=jnp.complex128) + 1j * jnp.cos(theta / 2))
        * generator
    )


def _controlled(operator: Array, n_control: int) -> Array:
    n_qubits = int(log2(operator.shape[0]))
    control = jnp.eye(2 ** (n_control + n_qubits), dtype=jnp.complex128)
    control = control.at[-(2**n_qubits) :, -(2**n_qubits) :].set(operator)
    return control


def product_state(bitstring: str) -> Array:
    """Generates a state of shape [2 for _ in range(len(bitstring))].

    Args:
        bitstring: The target bitstring.

    Returns:
        A state corresponding to 'bitstring'.
    """
    n_qubits = len(bitstring)
    space = jnp.zeros(tuple(2 for _ in range(n_qubits)), dtype=jnp.complex128)
    space = space.at[tuple(map(int, bitstring))].set(1.0)
    return space


def zero_state(n_qubits: int) -> Array:
    return product_state("0" * n_qubits)


def none_like(x: Iterable) -> Tuple[None, ...]:
    """Generates a tuple of Nones with equal length to x. Useful for gates with multiple targets but no control.

    Args:
        x (Iterable): Iterable to be mimicked.

    Returns:
        Tuple[None, ...]: Tuple of Nones of length x.
    """
    return tuple(map(lambda _: None, x))


def hilbert_reshape(operator: ArrayLike) -> Array:
    """Reshapes 'operator' of shape [M, M] to array of shape [2, 2, ...].
       Useful for working with controlled and multi-qubit gates.

    Args:
        operator (Array): Array to be reshaped.

    Returns:
        Array: Array with shape [2, 2, ...]
    """
    n_axes = int(np.log2(operator.size))
    return operator.reshape(tuple(2 for _ in np.arange(n_axes)))


def equivalent_state(s0: Array, s1: Array) -> bool:
    """Utility to easily compare two states.

    Args:
        s0: State array to be checked.
        s1:  State array to compare to.

    Returns:
        bool: Boolean indicating whether the states are equivalent.
    """
    return jnp.allclose(overlap(s0, s1), 1.0, atol=ATOL)  # type: ignore[no-any-return]


def inner(state: Array, projection: Array) -> Array:
    return jnp.dot(jnp.conj(state.flatten()), projection.flatten())


def overlap(state: Array, projection: Array) -> Array:
    return jnp.real(jnp.power(inner(state, projection), 2))


def uniform_state(
    n_qubits: int,
) -> Array:
    state = jnp.ones(2**n_qubits, dtype=jnp.complex128)
    state = state / jnp.sqrt(jnp.array(2**n_qubits, dtype=jnp.complex128))
    return state.reshape([2] * n_qubits)


def is_controlled(qs: Tuple[int | None, ...] | int | None) -> bool:
    if isinstance(qs, int):
        return True
    elif isinstance(qs, tuple):
        return any(is_controlled(q) for q in qs)
    return False


def random_state(n_qubits: int) -> Array:
    def _normalize(wf: Array) -> Array:
        return wf / jnp.sqrt((jnp.sum(jnp.abs(wf) ** 2)))

    key = jax.random.PRNGKey(n_qubits)
    N = 2**n_qubits
    x = -jnp.log(jax.random.uniform(key, shape=(N,)))
    sumx = jnp.sum(x)
    phases = jax.random.uniform(key, shape=(N,)) * 2.0 * jnp.pi
    return _normalize(
        (jnp.sqrt(x / sumx) * jnp.exp(1j * phases)).reshape(tuple(2 for _ in range(n_qubits)))
    )


def is_normalized(state: Array) -> bool:
    return equivalent_state(state, state)
