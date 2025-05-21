from __future__ import annotations

from functools import singledispatch
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify

from horqrux.apply import apply_gates
from horqrux.composite import Observable
from horqrux.primitives import Primitive
from horqrux.utils.operator_utils import DensityMatrix, State, expand_operator, num_qubits


@singledispatch
def eigen_probabilities(state: Any, eigvecs: Array) -> Array:
    """Obtain the probabilities using an input state and the eigenvectors decomposition
       of an observable.

    Args:
        state (Any): Input.
        eigvecs (Array): Eigenvectors of the observables.

    Returns:
        Array: The probabilities.
    """
    raise NotImplementedError(
        f"eigen_probabilities is not implemented for the state type {type(state)}."
    )


@eigen_probabilities.register
def _(state: Array, eigvecs: Array) -> Array:
    """Obtain the probabilities using an input quantum state vector
        and the eigenvectors decomposition
        of an observable.

    Args:
        state (Array): Input array.
        eigvecs (Array): Eigenvectors of the observables.

    Returns:
        Array: The probabilities.
    """
    inner_prod = jnp.matmul(jnp.conjugate(eigvecs.T), state.flatten())
    return jnp.abs(inner_prod) ** 2


@eigen_probabilities.register
def _(state: DensityMatrix, eigvecs: Array) -> Array:
    """Obtain the probabilities using an input quantum density matrix
        and the eigenvectors decomposition
        of an observable.

    Args:
        state (DensityMatrix): Input density matrix.
        eigvecs (Array): Eigenvectors of the observables.

    Returns:
        Array: The probabilities.
    """
    mat_prob = jnp.conjugate(eigvecs.T) @ state.array @ eigvecs
    return mat_prob.diagonal().real


def eigen_sample(
    state: State,
    observables: list[Observable],
    values: dict[str, float],
    n_qubits: int,
    n_shots: int,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """Sample eigenvalues of observable given the probability distribution
        defined by applying the eigenvectors to the state.

    Args:
        state (State): Input state or density matrix.
        observables (list[Observable]): list of observables.
        values (dict[str, float]): Parameter values.
        n_qubits (int): Number of qubits
        n_shots (int): Number of samples
        key (Any, optional): Random seed key. Defaults to jax.random.PRNGKey(0).

    Returns:
        Array: Sampled eigenvalues.
    """
    qubits = tuple(range(n_qubits))
    d = 2**n_qubits
    mat_obs = list(
        map(
            lambda observable: expand_operator(
                observable.tensor(values), observable.qubit_support, qubits
            ).reshape((d, d)),
            observables,
        )
    )
    eigs = jax.vmap(jnp.linalg.eigh)(jnp.stack(mat_obs))
    eigvecs, eigvals = align_eigenvectors(eigs.eigenvalues, eigs.eigenvectors)
    probs = eigen_probabilities(state, eigvecs)
    return jax.random.choice(key=key, a=eigvals, p=probs, shape=(n_shots,)).mean(axis=0)


@jax.vmap
def validate_permutation_matrix(P: Array) -> Array:
    rows = P.sum(axis=0)
    columns = P.sum(axis=1)
    ones = jnp.ones(P.shape[0], dtype=rows.dtype)
    return ((ones == rows) & (ones == columns)).min()


def checkify_valid_permutation(P: Array) -> None:
    checkify.check(
        jnp.all(validate_permutation_matrix(P)),
        "Did not calculate valid permutation matrix",
    )


def align_eigenvectors(eigenvalues: Array, eigenvectors: Array) -> tuple[Array, Array]:
    """
    Given a list of eigenvalue eigenvector matrix tuples in the form of
    [(eigenvalue, eigenvector)...], this function aligns all the eigenvector
    matrices so that they are identical, and also rearranges the corresponding
    eigenvalues.

    This is primarily used as a utility function to help sample multiple
    correlated observables when using finite shots.
    """
    eigenvector = eigenvectors[0]

    P = jax.vmap(lambda mat: permutation_matrix(mat, eigenvector))(eigenvectors)
    checkify.checkify(checkify_valid_permutation)(P)
    aligned_eigenvalues = jax.vmap(jnp.dot)(eigenvalues, P).T
    return eigenvector, aligned_eigenvalues


def permutation_matrix(mat: Array, eigenvector_matrix: Array) -> Array:
    """Obtain the permutation matrix for aligning eigenvectors.

    Given two permuted eigenvector matrices, A and B, we wish to find a permutation
    matrix P such that A P = B. This function calculates such a permutation
    matrix and uses it to align each eigenvector matrix to the first eigenvector
    matrix of eigs.

    Args:
        mat (Array): Matrix A.
        eigenvector_matrix (Array): Eigenvector matrix B.

    Returns:
        Array: Permutation matrix P.
    """
    return (jnp.linalg.inv(mat) @ eigenvector_matrix).real > 0.5


def finite_shots(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectations using `n_shots` shots per observable.

    Args:
        state (State): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.
        n_shots (int, optional): Number of shots. Defaults to 100.
        key (Any, optional): Key for randomness. Defaults to jax.random.PRNGKey(0).

    Returns:
        Array: Expectation values.
    """
    output_gates = apply_gates(state, gates, values)
    n_qubits = num_qubits(output_gates)
    if isinstance(state, DensityMatrix):
        d = 2**n_qubits
        output_gates.array = output_gates.array.reshape((d, d))
    return eigen_sample(output_gates, observables, values, n_qubits, n_shots, key)
