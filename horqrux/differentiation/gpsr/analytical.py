from __future__ import annotations

from functools import partial
from typing import Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from horqrux.apply import apply_gates
from horqrux.composite import Observable
from horqrux.differentiation.ad import _ad_expectation_single_observable
from horqrux.differentiation.gpsr.gpsr_utils import (
    extract_gate_names,
    initialize_gpsr_ingredients,
    prepare_param_gates_seq,
    spectral_gap_from_gates,
)
from horqrux.primitives import Primitive
from horqrux.utils.conversion import to_sparse
from horqrux.utils.operator_utils import State
from horqrux.utils.sparse_utils import stack_sp, sum_sp


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def no_shots_fwd(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectations analytically.

    This is compatible with `jax.grad`as we use the `jax.custom_jvp` for
    setting a custom differentiation rule to this function.

    Note though we de not recommend using `jax.jit` with it as compilation
    time is very long.

    Args:
        state (State): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.

    Returns:
        Array: Expectation values.
    """
    outputs = list(
        map(
            lambda observable: _ad_expectation_single_observable(
                apply_gates(state, gates, values),
                observable,
                values,
            ),
            observables,
        )
    )
    return stack_sp(outputs)


@no_shots_fwd.defjvp
def no_shots_fwd_jvp(
    state: Array,
    gates: Union[Primitive, Iterable[Primitive]],
    observable: list[Observable],
    primals: tuple[dict[str, float]],
    tangents: tuple[dict[str, float]],
) -> Array:
    """Jvp version for `no_shots_fwd`.

    Args:
        state (Array): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): sequence of gates.
        observable (list[Observable]): List of observables.
        primals (tuple[dict[str, Array]]): Values we are differentiating over.
            Jax-specific syntax argument.
        tangents (tuple[dict[str, Array]]): Bases for differentiation.
            Jax-specific syntax argument.

    Returns:
        tuple[Array, Array]: Forward eveluation and gradient
    """
    values = primals[0]
    tangent_dict = tangents[0]
    fwd = no_shots_fwd(state, gates, observable, values)

    val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)
    values_array = stack_sp(list(values.values()))

    def values_to_dict(x: Array) -> dict[str, Array]:
        return dict(zip(val_keys, x))

    def jvp_scan_norepeat(
        carry_grad: Array, pytree_ind_param: dict[str, Array]
    ) -> tuple[Array, Array]:
        shift_vector = pytree_ind_param["shift"]
        spectral_gap = pytree_ind_param["spectral_gap"]
        up_val = values_to_dict(values_array + shift_vector)
        f_up = no_shots_fwd(state, gates, observable, up_val)
        down_val = values_to_dict(values_array - shift_vector)
        f_down = no_shots_fwd(state, gates, observable, down_val)
        grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        return carry_grad + grad, grad

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
    init = jnp.zeros(fwd.shape)
    shifts = shift * jnp.eye(len(val_keys), dtype=values_array.dtype)
    if isinstance(fwd, BCOO):
        init = to_sparse(init)
        shifts = to_sparse(shifts)

    spectral_gap_array = stack_sp(list(spectral_gap.values()))
    tangent_array = stack_sp(list(tangent_dict.values())).reshape((-1, 1))
    pytree_scan = {
        "shift": shifts,
        "spectral_gap": spectral_gap_array,
    }

    _, grads = jax.lax.scan(jvp_caller, init, pytree_scan)
    jvp = sum_sp(stack_sp(grads) * tangent_array, axis=0)
    return fwd, jvp.reshape(fwd.shape)
