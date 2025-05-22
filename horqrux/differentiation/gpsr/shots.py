from __future__ import annotations

from functools import partial
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array, random
from jax.experimental.sparse import BCOO

from horqrux.composite import Observable
from horqrux.differentiation.gpsr.gpsr_utils import (
    extract_gate_names,
    initialize_gpsr_ingredients,
    prepare_param_gates_seq,
    spectral_gap_from_gates,
)
from horqrux.primitives import Primitive
from horqrux.shots import finite_shots
from horqrux.utils.conversion import to_sparse
from horqrux.utils.operator_utils import State
from horqrux.utils.sparse_utils import stack_sp, sum_sp


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 4, 5))
def finite_shots_fwd(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectations using `n_shots` shots per observable.

    This is compatible with `jax.grad`as we use the `jax.custom_jvp` for
    setting a custom differentiation rule to this function.

    Note though we de not recommend using `jax.jit` with it as compilation
    time is very long.

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
    return finite_shots(state, gates, observables, values, n_shots, key)


@finite_shots_fwd.defjvp
def finite_shots_jvp(
    state: Array,
    gates: Union[Primitive, Iterable[Primitive]],
    observable: list[Observable],
    n_shots: int,
    key: Array,
    primals: tuple[dict[str, Array]],
    tangents: tuple[dict[str, Array]],
) -> tuple[Array, Array]:
    """Jvp version for `finite_shots_fwd`.

    Args:
        state (Array): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): sequence of gates.
        observable (list[Observable]): List of observables.
        n_shots (int): Number of shots.
        key (Array): Key for randomness.
        primals (tuple[dict[str, Array]]): Values we are differentiating over.
            Jax-specific syntax argument.
        tangents (tuple[dict[str, Array]]): Bases for differentiation.
            Jax-specific syntax argument.

    Returns:
        tuple[Array, Array]: Forward eveluation and gradient
    """
    values = primals[0]
    tangent_dict = tangents[0]

    val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)
    values_array = stack_sp(list(values.values()))

    def values_to_dict(x: Array) -> dict[str, Array]:
        return dict(zip(val_keys, x))

    def jvp_scan_norepeat(carry_grad: Array, pytree_ind_param: dict[str, Array]) -> Array:
        key = pytree_ind_param["key"]
        up_key, down_key = random.split(key)
        shift_vector = pytree_ind_param["shift"]
        spectral_gap = pytree_ind_param["spectral_gap"]
        up_val = values_to_dict(values_array + shift_vector)
        f_up = finite_shots_fwd(state, gates, observable, up_val, n_shots, up_key)
        down_val = values_to_dict(values_array - shift_vector)
        f_down = finite_shots_fwd(state, gates, observable, down_val, n_shots, down_key)
        grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        return carry_grad + grad, grad

    fwd = finite_shots_fwd(state, gates, observable, values, n_shots, key)
    jvp_caller = jvp_scan_norepeat
    if not isinstance(gates, Primitive):
        gate_names = extract_gate_names(gates)
        if len(gate_names) > len(val_keys):
            param_to_gates_indices = prepare_param_gates_seq(val_keys, gates)
            # repeated case
            if max(map(len, param_to_gates_indices.values())) > 1:  # type: ignore[arg-type]
                raise NotImplementedError("Repeated case not implemented yet")

            else:
                spectral_gap = spectral_gap_from_gates(param_to_gates_indices, val_keys)

    spectral_gap_array = stack_sp(list(spectral_gap.values()))
    tangent_array = stack_sp(list(tangent_dict.values())).reshape((-1, 1))
    keys = random.split(key, len(values))

    init = jnp.zeros(fwd.shape)
    shifts = shift * jnp.eye(len(val_keys), dtype=values_array.dtype)
    if isinstance(fwd, BCOO):
        init = to_sparse(init)
        shifts = to_sparse(shifts)

    pytree_scan = {
        "shift": shifts,
        "spectral_gap": spectral_gap_array,
        "key": keys,
    }

    _, grads = jax.lax.scan(jvp_caller, init, pytree_scan)
    jvp = sum_sp(stack_sp(grads) * tangent_array, axis=0)
    return fwd, jvp.reshape(fwd.shape)
