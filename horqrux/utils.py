from __future__ import annotations

from enum import Enum
from functools import reduce
from math import log
from typing import Any, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy import log2

from ._misc import default_complex_dtype
from .matrices import _I

default_dtype = default_complex_dtype()

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
    """
    Create a controlled quantum operator with specified number of control qubits.

    Args:
        operator (jnp.ndarray): The base quantum operator to be controlled.
        n_control (int): Number of control qubits.

    Returns:
        jnp.ndarray: The controlled quantum operator matrix
    """
    n_qubits = int(log2(operator.shape[0]))
    control = jnp.eye(2 ** (n_control + n_qubits), dtype=default_dtype)
    control = control.at[-(2**n_qubits) :, -(2**n_qubits) :].set(operator)
    return control


def controlled(
    operator: jnp.ndarray,
    target_qubits: TargetQubits,
    control_qubits: ControlQubits,
) -> jnp.ndarray:
    """
    Create a controlled quantum operator with specified control and target qubit indices.

    Args:
        operator (jnp.ndarray): The base quantum operator to be controlled.
            Note the operator is defined only on `target_qubits`.
        control_qubits (int or tuple of ints): Index or indices of control qubits
        target_qubits (int or tuple of ints): Index or indices of target qubits

    Returns:
        jnp.ndarray: The controlled quantum operator matrix
    """
    controls: tuple = tuple()
    targets: tuple = tuple()
    if isinstance(control_qubits[0], tuple):
        controls = control_qubits[0]
    if isinstance(target_qubits[0], tuple):
        targets = target_qubits[0]
    nqop = int(log(operator.shape[0], 2))
    ntargets = len(targets)
    if nqop != ntargets:
        raise ValueError("`target_qubits` length should match the shape of operator.")
    # Determine the total number of qubits and order of controls
    ntotal_qubits = len(controls) + ntargets
    qubit_support = sorted(controls + targets)
    control_ind_support = tuple(i for i, q in enumerate(qubit_support) if q in controls)

    # Create the full Hilbert space dimension
    full_dim = 2**ntotal_qubits

    # Initialize the controlled operator as an identity matrix
    controlled_op = jnp.eye(full_dim, dtype=operator.dtype)

    # Compute the control mask using bit manipulation
    control_mask = jnp.sum(
        jnp.array(
            [1 << (ntotal_qubits - control_qubit - 1) for control_qubit in control_ind_support]
        )
    )

    # Create indices for the controlled subspace
    indices = jnp.arange(full_dim)
    controlled_indices = indices[(indices & control_mask) == control_mask]

    # Set the controlled subspace to the operator
    controlled_op = controlled_op.at[jnp.ix_(controlled_indices, controlled_indices)].set(operator)

    return controlled_op


def expand_operator(
    operator: Array, qubit_support: TargetQubits, full_support: TargetQubits
) -> Array:
    """
    Expands an operator acting on a given qubit_support to act on a larger full_support
    by explicitly filling in identity matrices on all remaining qubits.
    """
    full_support = tuple(sorted(full_support))
    qubit_support = tuple(sorted(qubit_support))
    if not set(qubit_support).issubset(set(full_support)):
        raise ValueError(
            "Expanding tensor operation requires a `full_support` argument "
            "larger than or equal to the `qubit_support`."
        )

    kron_qubits = set(full_support) - set(qubit_support)
    kron_operator = reduce(jnp.kron, [operator] + [_I] * len(kron_qubits))
    # TODO: Add permute_basis
    return kron_operator


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
    return jnp.power(inner(state, projection), 2).real


def uniform_state(
    n_qubits: int,
) -> Array:
    state = jnp.ones(2**n_qubits, dtype=default_dtype)
    state = state / jnp.sqrt(jnp.array(2**n_qubits, dtype=default_dtype))
    return state.reshape([2] * n_qubits)


def is_controlled(qubit_support: Tuple[int | None, ...] | int | None) -> bool:
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
