from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy import log2

from ._misc import default_complex_dtype

default_dtype = default_complex_dtype()

State = ArrayLike
QubitSupport = tuple[Any, ...]
ControlQubits = tuple[Union[None, tuple[int, ...]], ...]
TargetQubits = tuple[tuple[int, ...], ...]
ErrorProbabilities = Union[tuple[float, ...], float]

ATOL = 1e-014


def density_mat(state: Array) -> Array:
    """Convert state to density matrix

    Args:
        state (State): Input state.

    Returns:
        State: Density matrix representation.
    """
    # Expand dimensions to enable broadcasting
    ket = jnp.expand_dims(state, axis=tuple(range(state.ndim, 2 * state.ndim)))
    bra = jnp.conj(jnp.expand_dims(state, axis=tuple(range(state.ndim))))
    return ket * bra


def permute_basis(operator: Array, qubit_support: tuple, inv: bool = False) -> Array:
    """Takes an operator tensor and permutes the rows and
    columns according to the order of the qubit support.

    Args:
        operator (Tensor): Operator to permute over.
        qubit_support (tuple): Qubit support.
        inv (bool): Applies the inverse permutation instead.

    Returns:
        Tensor: Permuted operator.
    """
    ordered_support = np.argsort(qubit_support)
    ranked_support = np.argsort(ordered_support)
    n_qubits = len(qubit_support)
    if all(a == b for a, b in zip(ranked_support, tuple(range(n_qubits)))):
        return operator

    perm = tuple(ranked_support) + tuple(ranked_support + n_qubits)
    if inv:
        perm = np.argsort(perm)
    return jax.lax.transpose(operator, perm)


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


class DiffMode(StrEnum):
    """Differentiation mode."""

    AD = "ad"
    """Automatic Differentiation -  Using the autograd engine of JAX."""
    ADJOINT = "adjoint"
    """Adjoint Differentiation   - An implementation of "Efficient calculation of gradients
                                   in classical simulations of variational quantum algorithms",
                                   Jones & Gacon, 2020."""
    GPSR = "gpsr"
    """Generalized parameter shift rule."""


class ForwardMode(StrEnum):
    EXACT = "exact"
    SHOTS = "shots"


def _dagger(operator: Array) -> Array:
    # If the operator is a tensor with repeated 2D axes
    if operator.ndim > 2:
        # Conjugate and swap the last two axes
        conjugated = operator.conj()

        # Create the transpose axes: swap pairs of indices
        half = operator.ndim // 2
        axes = tuple(range(half, operator.ndim)) + tuple(range(half))
        return jnp.transpose(conjugated, axes)
    else:
        # For standard matrices, use conjugate transpose
        return jnp.conjugate(operator.T)


def _unitary(generator: Array, theta: float) -> Array:
    return (
        jnp.cos(theta / 2) * jnp.eye(2, dtype=default_dtype) - 1j * jnp.sin(theta / 2) * generator
    )


def _jacobian(generator: Array, theta: float) -> Array:
    return (
        -1
        / 2
        * (jnp.sin(theta / 2) * jnp.eye(2, dtype=default_dtype) + 1j * jnp.cos(theta / 2))
        * generator
    )


def _controlled(operator: Array, n_control: int) -> Array:
    n_qubits = int(log2(operator.shape[0]))
    control = jnp.eye(2 ** (n_control + n_qubits), dtype=default_dtype)
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
    space = jnp.zeros(tuple(2 for _ in range(n_qubits)), dtype=default_dtype)
    space = space.at[tuple(map(int, bitstring))].set(1.0)
    return space


def zero_state(n_qubits: int) -> Array:
    return product_state("0" * n_qubits)


def none_like(x: Iterable) -> tuple[None, ...]:
    """Generates a tuple of Nones with equal length to x. Useful for gates with multiple targets but no control.

    Args:
        x (Iterable): Iterable to be mimicked.

    Returns:
        tuple[None, ...]: tuple of Nones of length x.
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
    return jnp.power(inner(state, projection), 2).real


def uniform_state(
    n_qubits: int,
) -> Array:
    state = jnp.ones(2**n_qubits, dtype=default_dtype)
    state = state / jnp.sqrt(jnp.array(2**n_qubits, dtype=default_dtype))
    return state.reshape([2] * n_qubits)


def is_controlled(qubit_support: tuple[int | None, ...] | int | None) -> bool:
    if isinstance(qubit_support, int):
        return True
    elif isinstance(qubit_support, tuple):
        return any(is_controlled(q) for q in qubit_support)
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
