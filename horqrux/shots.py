from __future__ import annotations

from functools import partial, reduce
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, random
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


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 4, 5))
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
