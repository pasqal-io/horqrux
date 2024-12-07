from __future__ import annotations

from functools import partial, reduce
from typing import Any
from jax.custom_derivatives import SymbolicZero

import jax
import jax.numpy as jnp
from jax import Array, random
from jax import lax

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


# @jax.custom_jvp
@partial(jax.custom_jvp, nondiff_argnums=(4,))
def finite_shots_fwd(
    state: Array,
    gates: GateSequence,
    observables: list[Primitive],
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
    shift_up_gates=jnp.array([], dtype=int),
    shift_down_gates=jnp.array([], dtype=int),
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    state = apply_gate(state=state, gate=gates, values=values, shift_up_gates=shift_up_gates,
                       shift_down_gates=shift_down_gates)
    n_qubits = len(state.shape)
    mat_obs = [observable_to_matrix(observable, n_qubits) for observable in observables]
    eigs = [jnp.linalg.eigh(mat) for mat in mat_obs]
    eigvecs, eigvals = align_eigenvectors(eigs)
    inner_prod = jnp.matmul(jnp.conjugate(eigvecs.T), state.flatten())
    probs = jnp.abs(inner_prod) ** 2
    return jax.random.choice(key, eigvals, (n_shots,), True, probs).mean(axis=0)


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


def get_shifted_gate(gate, values, shift):
    current_angle = gate.parse_values(values)
    new_angle = current_angle + shift
    return Parametric(gate.generator_name, gate.target[0], gate.control[0], new_angle)


@partial(finite_shots_fwd.defjvp, symbolic_zeros=True)
def finite_shots_jvp(
        n_shots,
        primals: tuple[list[Primitive], dict[str, float]],
        tangents: tuple[list[Primitive], dict[str, float]]
) -> Array:

    # state: Array,
    # gates: GateSequence,
    # observables: list[Primitive],
    # values: dict[str, float],
    # n_shots: int = 100,
    # key: Any = jax.random.PRNGKey(0),
    # shift_up_gates=jnp.array([], dtype=int),
    # shift_down_gates=jnp.array([], dtype=int),

    state, gates, observables, values,  key, shift_up_gates, shift_down_gates = primals
    fwd = finite_shots_fwd(state, gates, observables, values, n_shots,
                           key, shift_up_gates, shift_down_gates)
    zero = jnp.zeros_like(fwd)
    jvp = jnp.zeros_like(fwd)

    gate_tangents = [gate.param if isinstance(
        gate, Parametric) else None for gate in tangents[1]]
    gate_tangents = [tangents[3][param] if isinstance(
        param, str) else param for param in gate_tangents]
    gate_tangents = [tangent if not isinstance(
        tangent, SymbolicZero) else None for tangent in gate_tangents]
    gate_tangents = [tangent if isinstance(
        tangent, jax.Array) else None for tangent in gate_tangents]

    def parametric_gradient_at_i(i, primals, n_shots):
        state, gates, observables, values, key, shift_up_gates, shift_down_gates = primals
        base_key = random.split(key, len(gates))[i]
        up_key, down_key = random.split(base_key, 2)
        new_shift_up_gates = jnp.append(shift_up_gates, i)
        new_shift_down_gates = jnp.append(shift_down_gates, i)
        f_up = finite_shots_fwd(state, gates, observables, values, n_shots,
                                up_key, new_shift_up_gates, shift_down_gates)
        f_down = finite_shots_fwd(state, gates, observables, values, n_shots,
                                  down_key, shift_up_gates, new_shift_down_gates)
        shift = jnp.pi/2
        spectral_gap = 2.0
        return spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))

    # def loop_func(i, carry):
    #    primals, zero, gate_tangents, jvp = carry
    #    jvp_component = lax.cond(jnp.isnan(gate_tangents[i]),
    #                             lambda *args: zero,
    #                             lambda i, primals, tangent: tangent[i] *
    #                             parametric_gradient_at_i(i, primals),
    #                             i,
    #                             primals,
    #                             gate_tangents,
    #                             )
    #    return primals, zero, gate_tangents, jvp + jvp_component

    for i, _ in enumerate(gates):
        if gate_tangents[i] is None:
            continue
        jvp = jvp + gate_tangents[i] * parametric_gradient_at_i(i, primals, n_shots)

    # init_carry = primals, zero, gate_tangents, jvp
    # loop_out = lax.fori_loop(0, gate_tangents.shape[0], loop_func, init_carry)

    return fwd, jvp
