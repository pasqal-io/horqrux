from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import reduce, singledispatch
from math import log
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse import BCOO, sparsify
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from numpy import log2

from horqrux._misc import default_complex_dtype
from horqrux.matrices import _I

from .sparse_utils import kron_sp

default_dtype = default_complex_dtype()

QubitSupport = tuple[Any, ...]
ControlQubits = tuple[Union[None, tuple[int, ...]], ...]
TargetQubits = tuple[tuple[int, ...], ...]
ATOL = 1e-014


@register_pytree_node_class
@dataclass
class DensityMatrix:
    """Dataclass to identify density matrices from states."""

    array: Array

    def tree_flatten(self) -> tuple[tuple, tuple[Array]]:
        children = ()
        aux_data = (self.array,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)


State = Union[ArrayLike, DensityMatrix]


def density_mat(state: ArrayLike) -> DensityMatrix:
    """Convert state to density matrix

    Args:
        state (ArrayLike): Input state.

    Returns:
        DensityMatrix: Density matrix representation.
    """
    # Expand dimensions to enable broadcasting
    if isinstance(state, DensityMatrix):
        return state
    ket = jnp.expand_dims(state, axis=tuple(range(state.ndim, 2 * state.ndim)))
    bra = jnp.conj(jnp.expand_dims(state, axis=tuple(range(state.ndim))))
    return DensityMatrix(ket * bra)


def permute_basis(operator: Array | BCOO, qubit_support: tuple, inv: bool = False) -> Array | BCOO:
    """Takes an operator tensor and permutes the rows and
    columns according to the order of the qubit support.

    Args:
        operator (Array | BCOO): Operator to permute over.
        qubit_support (tuple): Qubit support.
        inv (bool): Applies the inverse permutation instead.

    Returns:
        Array | BCOO: Permuted operator.
    """
    ordered_support = np.argsort(qubit_support)
    ranked_support = np.argsort(ordered_support)
    n_qubits = len(qubit_support)
    if all(a == b for a, b in zip(ranked_support, tuple(range(n_qubits)))):
        return operator

    perm = tuple(ranked_support) + tuple(ranked_support + n_qubits)
    if inv:
        perm = np.argsort(perm)
    transpose_fn = (
        jax.experimental.sparse.bcoo_transpose if isinstance(operator, BCOO) else jax.lax.transpose
    )
    return transpose_fn(operator, permutation=perm)


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class OperationType(StrEnum):
    UNITARY = "_unitary"
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


@sparsify
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


@singledispatch
def _unitary(generator: Any, theta: float) -> Any:
    raise NotImplementedError("_unitary is not implemented")


@_unitary.register
def _(generator: Array, theta: float) -> Array:
    return (
        jnp.cos(theta / 2) * jnp.eye(2, dtype=default_dtype) - 1j * jnp.sin(theta / 2) * generator
    )


@_unitary.register
def _(generator: BCOO, theta: float) -> BCOO:
    return (
        jnp.cos(theta / 2) * jax.experimental.sparse.eye(2, dtype=default_dtype)
        - 1j * jnp.sin(theta / 2) * generator
    )


@singledispatch
def _jacobian(generator: Any, theta: float) -> Any:
    raise NotImplementedError("_jacobian is not implemented")


@_jacobian.register
def _(generator: Array, theta: float) -> Array:
    return (
        -1
        / 2
        * (jnp.sin(theta / 2) * jnp.eye(2, dtype=default_dtype) + 1j * jnp.cos(theta / 2))
        * generator
    )


@_jacobian.register
def _(generator: BCOO, theta: float) -> BCOO:
    return (
        -1
        / 2
        * (
            jnp.sin(theta / 2) * jax.experimental.sparse.eye(2, dtype=default_dtype)
            + 1j * jnp.cos(theta / 2)
        )
        * generator
    )


@singledispatch
def _controlled(operator: Any, n_control: int) -> Any:
    raise NotImplementedError("_controlled is not implemented")


@_controlled.register
def _(operator: Array, n_control: int) -> Array:
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


@_controlled.register
def _(operator: BCOO, n_control: int) -> BCOO:
    """
    Create a controlled quantum operator with specified number of control qubits.

    Args:
        operator (jnp.ndarray): The base quantum operator to be controlled.
        n_control (int): Number of control qubits.

    Returns:
        jnp.ndarray: The controlled quantum operator matrix
    """
    n_qubits = int(log2(operator.shape[0]))
    shape_dim = 2 ** (n_control + n_qubits)
    control_elements = shape_dim - 2**n_qubits
    control_id = jax.experimental.sparse.eye(control_elements, dtype=default_dtype)
    data_control = jnp.concatenate((control_id.data, operator.data))
    indices_control = jnp.concatenate((control_id.indices, operator.indices + control_elements))
    control = jax.experimental.sparse.BCOO(
        (data_control, indices_control), shape=(shape_dim, shape_dim)
    )
    return control


@singledispatch
def controlled(
    operator: Any,
    target_qubits: TargetQubits,
    control_qubits: ControlQubits,
) -> Any:
    raise NotImplementedError(
        f"Controlled is not implemented for this operator type: {type(operator)}."
    )


def prepare_controlled(
    operator: ArrayLike, target_qubits: TargetQubits, control_qubits: ControlQubits
) -> tuple[int, ArrayLike]:
    controls: tuple = tuple()
    targets: tuple = tuple()
    if isinstance(control_qubits[0], tuple):
        controls = control_qubits[0]
    if isinstance(target_qubits[0], tuple):
        targets = target_qubits[0]
    n_qop = int(log(operator.shape[0], 2))
    n_targets = len(targets)
    if n_qop != n_targets:
        raise ValueError("`target_qubits` length should match the shape of operator.")
    # Determine the total number of qubits and order of controls
    ntotal_qubits = len(controls) + n_targets
    qubit_support = sorted(controls + targets)
    control_ind_support = tuple(i for i, q in enumerate(qubit_support) if q in controls)

    # Create the full Hilbert space dimension
    full_dim = 2**ntotal_qubits

    # Compute the control mask using bit manipulation
    control_mask = jnp.sum(
        jnp.array(
            [1 << (ntotal_qubits - control_qubit - 1) for control_qubit in control_ind_support]
        )
    )

    # Create indices for the controlled subspace
    indices = jnp.arange(full_dim)
    controlled_indices = indices[(indices & control_mask) == control_mask]
    return full_dim, controlled_indices


@controlled.register
def _(
    operator: Array,
    target_qubits: TargetQubits,
    control_qubits: ControlQubits,
) -> Array:
    """
    Create a controlled quantum operator with specified control and target qubit indices.

    Args:
        operator (Array): The base quantum operator to be controlled.
            Note the operator is defined only on `target_qubits`.
        control_qubits (int or tuple of ints): Index or indices of control qubits
        target_qubits (int or tuple of ints): Index or indices of target qubits

    Returns:
        Array: The controlled quantum operator matrix
    """
    full_dim, controlled_indices = prepare_controlled(operator, target_qubits, control_qubits)
    # Initialize the controlled operator as an identity matrix
    # and set the controlled subspace to the operator
    controlled_op = jnp.eye(full_dim, dtype=operator.dtype)
    controlled_op = controlled_op.at[jnp.ix_(controlled_indices, controlled_indices)].set(operator)

    return controlled_op


@controlled.register
def _(
    operator: BCOO,
    target_qubits: TargetQubits,
    control_qubits: ControlQubits,
) -> BCOO:
    """
    Create a controlled quantum operator with specified control and target qubit indices.

    Args:
        operator (Array): The base quantum operator to be controlled.
            Note the operator is defined only on `target_qubits`.
        control_qubits (int or tuple of ints): Index or indices of control qubits
        target_qubits (int or tuple of ints): Index or indices of target qubits

    Returns:
        Array: The controlled quantum operator matrix
    """
    full_dim, controlled_indices = prepare_controlled(operator, target_qubits, control_qubits)

    # Initialize the controlled operator as an identity matrix
    # and set the controlled subspace to the operator
    # TODO: optimize this
    controlled_op = jnp.eye(full_dim, dtype=operator.dtype)
    controlled_op = controlled_op.at[jnp.ix_(controlled_indices, controlled_indices)].set(
        operator.todense()
    )
    return BCOO.fromdense(controlled_op)


def expand_operator(
    operator: Array | BCOO, qubit_support: tuple[int, ...], full_support: tuple[int, ...]
) -> Array | BCOO:
    """
    Expands an operator acting on a given qubit_support to act on a larger full_support
    by explicitly filling in identity matrices on all remaining qubits.

    Args:
        operator (Array | BCOO): Operator to expand
        qubit_support (tuple[int, ...]): Qubit support the operator is initially defined over.
        full_support (tuple[int, ...]): Qubit support the operator will be defined over.

    Raises:
        ValueError: When `full_support` larger than or equal to the `qubit_support`

    Returns:
        Array | BCOO: Expanded operator.
    """
    full_support = tuple(sorted(full_support))
    qubit_support = tuple(sorted(qubit_support))
    n_full = 2 ** len(full_support)
    if not set(qubit_support).issubset(set(full_support)):
        raise ValueError(
            "Expanding tensor operation requires a `full_support` argument "
            "larger than or equal to the `qubit_support`."
        )

    kron_qubits = tuple(sorted(set(full_support) - set(qubit_support)))
    if isinstance(operator, BCOO):
        kron_operator = reduce(
            kron_sp,
            [operator] + [jax.experimental.sparse.eye(2, dtype=default_dtype)] * len(kron_qubits),
        )
    else:
        kron_operator = reduce(jnp.kron, [operator] + [_I] * len(kron_qubits))

    kron_operator = hilbert_reshape(kron_operator)
    kron_operator = permute_basis(kron_operator, qubit_support + kron_qubits, True)
    kron_operator = kron_operator.reshape((n_full, n_full))
    return kron_operator


def product_state(bitstring: str, sparse: bool = False) -> Array:
    """Generates a state of shape [2 for _ in range(len(bitstring))].

    Args:
        bitstring: The target bitstring.

    Returns:
        A state corresponding to 'bitstring'.
    """
    n_qubits = len(bitstring)
    shape = tuple(2 for _ in range(n_qubits))
    bitstr_ind = tuple(map(int, bitstring))
    if sparse:
        data = jnp.array([1.0], dtype=default_dtype)
        indices = jnp.array([bitstr_ind])
        space = BCOO((data, indices), shape=shape)
    else:
        space = jnp.zeros(shape, dtype=default_dtype)
        space = space.at[bitstr_ind].set(1.0)
    return space


def zero_state(n_qubits: int, sparse: bool = False) -> Array:
    return product_state("0" * n_qubits, sparse)


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


@sparsify
def inner(state: Array, projection: Array) -> Array:
    return jnp.dot(jnp.conj(state.flatten()), projection.flatten())


def overlap(state: Array, projection: Array) -> Array:
    return jnp.power(inner(state, projection), 2).real


def uniform_state(n_qubits: int, sparse: bool = False) -> ArrayLike:
    state = jnp.ones(2**n_qubits, dtype=default_dtype)
    state = state / jnp.sqrt(jnp.array(2**n_qubits, dtype=default_dtype))
    if sparse:
        state = BCOO.fromdense(state)
    return state.reshape([2] * n_qubits)


def is_controlled(qubit_support: Union[tuple[Union[int, None], ...], int, None]) -> bool:
    if isinstance(qubit_support, int):
        return True
    elif isinstance(qubit_support, tuple):
        return any(is_controlled(q) for q in qubit_support)
    return False


def random_state(n_qubits: int, sparse: bool = False) -> ArrayLike:
    def _normalize(wf: Array) -> Array:
        return wf / jnp.sqrt((jnp.sum(jnp.abs(wf) ** 2)))

    key = jax.random.PRNGKey(n_qubits)
    N = 2**n_qubits
    x = -jnp.log(jax.random.uniform(key, shape=(N,)))
    sumx = jnp.sum(x)
    phases = jax.random.uniform(key, shape=(N,)) * 2.0 * jnp.pi
    state = _normalize(
        (jnp.sqrt(x / sumx) * jnp.exp(1j * phases)).reshape(tuple(2 for _ in range(n_qubits)))
    )
    if sparse:
        state = BCOO.fromdense(state)
    return state


def is_normalized(state: Array) -> bool:
    return equivalent_state(state, state)


def sample_from_probs(probs: Array, n_qubits: int, n_shots: int) -> Counter:
    key = jax.random.PRNGKey(0)

    # JAX handles pseudo random number generation by tracking an explicit state via a random key
    # For more details, see https://jax.readthedocs.io/en/latest/random-numbers.html
    samples = jax.vmap(
        lambda subkey: jax.random.choice(key=subkey, a=jnp.arange(0, 2**n_qubits), p=probs)
    )(jax.random.split(key, n_shots))

    return Counter(
        {
            format(k, f"0{n_qubits}b"): count.item()
            for k, count in enumerate(jnp.bincount(samples))
            if count > 0
        }
    )


@singledispatch
def probabilities(state: Any) -> Array:
    """Extract probabilities from state or density matrix.

    Args:
        state (Array): Input array.

    Raises:
        NotImplementedError: If not implemented for given types.

    Returns:
        Array: Vector of probabilities.
    """
    raise NotImplementedError(f"Probabilities is not implemented for the input type {type(state)}.")


@probabilities.register
def _(state: Array) -> Array:
    return jnp.abs(jnp.float_power(state, 2.0)).ravel()


@probabilities.register
def _(state: DensityMatrix) -> Array:
    return jnp.diagonal(state.array).real


@singledispatch
def num_qubits(state: Any) -> int:
    """Returns the number of qubits of a state.

    Args:
        state (Any): state.

    Returns:
        int: Number of qubits.
    """
    raise NotImplementedError(f"num_qubits is not implemented for the state type {type(state)}.")


@num_qubits.register
def _(state: Array) -> int:
    return len(state.shape)


@num_qubits.register
def _(state: DensityMatrix) -> int:
    return len(state.array.shape) // 2
