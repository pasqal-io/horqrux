from __future__ import annotations

from functools import partial, reduce
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, random

from horqrux.apply import apply_gate
from horqrux.parametric import Parametric
from horqrux.primitive import GateSequence, Primitive


def observable_to_matrix(observable: Primitive, n_qubits: int) -> Array:
    """For finite shot sampling we need to calculate the eigenvalues/vectors of
    an observable. This helper function takes an observable and system size
    (n_qubits) and returns the overall action of the observable on the whole
    system.

    LIMITATION: currently only works for observables which are not controlled.
    """
    unitary = observable.unitary()
    target = observable.target[0][0]
    identity = jnp.eye(2, dtype=unitary.dtype)
    ops = [identity for _ in range(n_qubits)]
    ops[target] = unitary
    return reduce(lambda x, y: jnp.kron(x, y), ops[1:], ops[0])


@partial(jax.custom_jvp, nondiff_argnums=(0, 2, 4, 5))
def finite_shots_fwd(
    state: Array,
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
    state = apply_gate(state, gates, values)
    n_qubits = len(state.shape)
    mat_obs = [observable_to_matrix(observable, n_qubits) for observable in observables]
    eigs = [jnp.linalg.eigh(mat) for mat in mat_obs]
    eigvecs, eigvals = align_eigenvectors(eigs)
    inner_prod = jnp.matmul(jnp.conjugate(eigvecs.T), state.flatten())
    probs = jnp.abs(inner_prod) ** 2
    return jax.random.choice(key=key, a=eigvals, p=probs, shape=(n_shots,)).mean(axis=0)


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
    observable: Primitive,
    n_shots: int,
    key: Array,
    primals: tuple[list[Primitive], dict[str, float]],
    tangents: tuple[list[Primitive], dict[str, float]],
) -> Array:
    gates, values = primals
    gates_tangent, values_tangent = tangents

    # TODO: compute spectral gap through the generator which is associated with
    # a param name.
    spectral_gap = 2.0
    shift = jnp.pi / 2

    fwd = finite_shots_fwd(state, gates, observable, values, n_shots, key)

    keys = random.split(key, len(gates))
    jvp = jnp.zeros_like(fwd)
    zero = jnp.zeros_like(fwd)

    def jvp_component(index: int) -> Array:
        gates, values = primals
        gates_tangent, values_tangent = tangents
        shift_gate = gates[index]
        gate_tangent = gates_tangent[index]
        if not isinstance(shift_gate, Parametric) or not isinstance(gate_tangent, Parametric):
            return zero
        if shift_gate.param_name is None:
            tangent = gate_tangent.param_val
        else:
            tangent = values_tangent[shift_gate.param_name]
        if not isinstance(tangent, jax.Array):
            return zero
        key = keys[index]
        up_key, down_key = random.split(key)
        original_shift = shift_gate.shift
        shift_gate.set_shift(original_shift + shift)
        f_up = finite_shots_fwd(state, gates, observable, values, n_shots, up_key)
        shift_gate.set_shift(original_shift - shift)
        f_down = finite_shots_fwd(state, gates, observable, values, n_shots, down_key)
        shift_gate.set_shift(original_shift)
        grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        return grad * tangent

    return fwd, sum(jvp_component(i) for i, _ in enumerate(gates))
