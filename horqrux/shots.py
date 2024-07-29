from __future__ import annotations

from functools import reduce
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify

from horqrux.apply import apply_gate
from horqrux.primitive import GateSequence, Primitive
from horqrux.utils import none_like


def observable_to_matrix(observable: Primitive, n_qubits: int) -> Array:
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
    unitary = observable.unitary()
    target = observable.target[0][0]
    identity = jnp.eye(2, dtype=unitary.dtype)
    ops = [identity for _ in range(n_qubits)]
    ops[target] = unitary
    return reduce(lambda x, y: jnp.kron(x, y), ops[1:], ops[0])


def finite_shots_fwd(
    state: Array,
    gates: GateSequence,
    observable: Primitive,
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
    mat_obs = observable_to_matrix(observable, n_qubits)
    eigvals, eigvecs = jnp.linalg.eigh(mat_obs)
    inner_prod = jnp.matmul(jnp.conjugate(eigvecs.T), state.flatten())
    probs = jnp.abs(inner_prod) ** 2
    return jax.random.choice(key=key, a=eigvals, p=probs, shape=(n_shots,)).mean()
