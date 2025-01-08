from __future__ import annotations

from functools import partial, reduce, singledispatch
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, random
from jax.experimental import checkify

from horqrux.apply import apply_gate
from horqrux.primitive import GateSequence, Primitive
from horqrux.utils import DensityMatrix, State, none_like


def to_matrix(
    observable: Primitive,
    n_qubits: int,
    values: dict[str, float],
) -> Array:
    """For finite shot sampling we need to calculate the eigenvalues/vectors of
    an observable. This helper function takes an observable and system size
    (n_qubits) and returns the overall action of the observable on the whole
    system.

    LIMITATION: currently only works for observables which are not controlled.
    """
    checkify.check(
        observable.control == observable.parse_idx(none_like(observable.target)),
        "Controlled gates cannot be promoted from observables to operations on the whole state vector",
    )
    unitary = observable.unitary(values=values)
    target = observable.target[0][0]
    identity = jnp.eye(2, dtype=unitary.dtype)
    ops = [identity for _ in range(n_qubits)]
    ops[target] = unitary
    return reduce(lambda x, y: jnp.kron(x, y), ops[1:], ops[0])


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
    observables: list[Primitive],
    values: dict[str, float],
    n_qubits: int,
    n_shots: int,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    mat_obs = [to_matrix(observable, n_qubits, values) for observable in observables]
    eigs = [jnp.linalg.eigh(mat) for mat in mat_obs]
    eigvecs, eigvals = align_eigenvectors(eigs)
    probs = eigen_probabilities(state, eigvecs)
    return jax.random.choice(key=key, a=eigvals, p=probs, shape=(n_shots,)).mean(axis=0)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 4, 5))
def finite_shots_fwd(
    state: State,
    gates: GateSequence,
    observables: list[Primitive],
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    if isinstance(state, DensityMatrix):
        output_gates = apply_gate(state, gates, values)
        n_qubits = len(output_gates.array.shape) // 2
        d = 2**n_qubits
        output_gates.array = output_gates.array.reshape((d, d))
    else:
        output_gates = apply_gate(state, gates, values)
        n_qubits = len(state.shape)
    return eigen_sample(
        output_gates, observables, values, n_qubits, n_shots, key
    )


def align_eigenvectors(eigs: list[tuple[Array, Array]]) -> tuple[Array, Array]:
    """
    Given a list of eigenvalue eigenvector matrix tuples in the form of
    [(eigenvalue, eigenvector)...], this function aligns all the eigenvector
    matrices so that they are identical, and also rearranges the corresponding
    eigenvalues.

    This is primarily used as a utility function to help sample multiple
    correlated observables when using finite shots.

    Given two permuted eigenvector matrices, A and B, we wish to find a permutation
    matrix P such that A P = B. This function calculates such a permutation
    matrix and uses it to align each eigenvector matrix to the first eigenvector
    matrix of eigs.
    """
    eigenvalues = []
    eigs_copy = eigs.copy()
    eigenvalue, eigenvector_matrix = eigs_copy.pop(0)
    eigenvalues.append(eigenvalue)
    # TODO: laxify this loop
    for mat in eigs_copy:
        inv = jnp.linalg.inv(mat[1])
        P = (inv @ eigenvector_matrix).real > 0.5
        checkify.check(
            validate_permutation_matrix(P),
            "Did not calculate valid permutation matrix",
        )
        eigenvalues.append(mat[0] @ P)
    return eigenvector_matrix, jnp.stack(eigenvalues, axis=1)


def validate_permutation_matrix(P: Array) -> Array:
    rows = P.sum(axis=0)
    columns = P.sum(axis=1)
    ones = jnp.ones(P.shape[0], dtype=rows.dtype)
    return ((ones == rows) & (ones == columns)).min()


@finite_shots_fwd.defjvp
def finite_shots_jvp(
    state: Array,
    gates: GateSequence,
    observable: Primitive,
    n_shots: int,
    key: Array,
    primals: tuple[dict[str, float]],
    tangents: tuple[dict[str, float]],
) -> Array:
    values = primals[0]
    tangent_dict = tangents[0]

    # TODO: compute spectral gap through the generator which is associated with
    # a param name.
    spectral_gap = 2.0
    shift = jnp.pi / 2

    def jvp_component(param_name: str, key: Array) -> Array:
        up_key, down_key = random.split(key)
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = finite_shots_fwd(state, gates, observable, up_val, n_shots, up_key)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = finite_shots_fwd(state, gates, observable, down_val, n_shots, down_key)
        grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        return grad * tangent_dict[param_name]

    params_with_keys = zip(values.keys(), random.split(key, len(values)))
    fwd = finite_shots_fwd(state, gates, observable, values, n_shots, key)
    jvp = sum(jvp_component(param, key) for param, key in params_with_keys)
    return fwd, jvp.reshape(fwd.shape)
