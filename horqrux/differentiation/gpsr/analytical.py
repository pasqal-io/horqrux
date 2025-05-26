from __future__ import annotations

from functools import partial
from typing import Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array

from horqrux.apply import apply_gates
from horqrux.composite import Observable
from horqrux.differentiation.ad import _ad_expectation_single_observable
from horqrux.differentiation.gpsr.gpsr_utils import (
    alter_gate_sequence,
    create_renamed_operators,
    extract_gate_names,
    initialize_gpsr_ingredients,
    prepare_param_gates_seq,
    spectral_gap_from_gates,
)
from horqrux.primitives import Primitive
from horqrux.utils.operator_utils import State
from horqrux.utils.sparse_utils import stack_sp


def analytical_expectation(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectations analytically.

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


jitted_analytical_exp = jax.jit(analytical_expectation)


def analytical_gpsr_bwd(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
) -> Array:
    """Compute the gradient using the GPSR algorithm.
    This function is efficient when used in a
    training loop using the jitted version
    of the `no_shots` function
    (that is the `horqrux.differentiation.gpsr.jitted_analytical_exp`).

    There is also a jvp function `no_shots_fwd`compatible with `jax.grad`
    but this function may be faster when using `jax.jit`.

    Args:
        state (State): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.

    Returns:
        Array: Gradients via the GPSR algorithm.
    """
    val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)

    def jvp_component(param_name: str) -> Array:
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = jitted_analytical_exp(state, gates, observables, up_val)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = jitted_analytical_exp(state, gates, observables, down_val)
        grad = (
            spectral_gap[param_name]
            * (f_up - f_down)
            / (4.0 * jnp.sin(spectral_gap[param_name] * shift / 2.0))
        )
        return grad

    jvp_caller = jvp_component
    if not isinstance(gates, Primitive):
        gate_names = extract_gate_names(gates)
        if len(gate_names) > len(val_keys):
            param_to_gates_indices = prepare_param_gates_seq(val_keys, gates)
            # repeated case
            if max(map(len, param_to_gates_indices.values())) > 1:  # type: ignore[arg-type]

                def jvp_component_repeated_param(param_name: str) -> Array:
                    shift_gates = param_to_gates_indices[param_name]

                    def shift_jvp(ind: int) -> Array:
                        spectral_gap = gates[ind].spectral_gap  # type: ignore[index]
                        gates_up = alter_gate_sequence(gates, ind, shift)
                        f_up = jitted_analytical_exp(state, gates_up, observables, values)
                        gates_down = alter_gate_sequence(gates, ind, -shift)
                        f_down = jitted_analytical_exp(state, gates_down, observables, values)
                        return (
                            spectral_gap
                            * (f_up - f_down)
                            / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                        )

                    return sum(shift_jvp(shift_ind) for shift_ind in shift_gates)

                jvp_caller = jvp_component_repeated_param
            else:
                spectral_gap = spectral_gap_from_gates(param_to_gates_indices, val_keys)

    jvp = jnp.stack([jvp_caller(param) for param in val_keys]).sum(axis=-1)
    return jvp


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def analytical_gpsr_fwd(
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
    return analytical_expectation(state, gates, observables, values)


@analytical_gpsr_fwd.defjvp
def analytical_gpsr_jvp(
    state: Array,
    gates: Union[Primitive, Iterable[Primitive]],
    observable: list[Observable],
    primals: tuple[dict[str, float]],
    tangents: tuple[dict[str, float]],
) -> Array:
    """Jvp version for `analytical_gpsr_fwd`.

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
    fwd = analytical_gpsr_fwd(state, gates, observable, values)

    legit_val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)
    values_map = None

    if not isinstance(gates, Primitive):
        gate_names = extract_gate_names(gates)
        if len(gate_names) > len(legit_val_keys):
            param_to_gates_indices = prepare_param_gates_seq(legit_val_keys, gates)
            # repeated case
            if max(map(len, param_to_gates_indices.values())) > 1:  # type: ignore[arg-type]
                # use temporary gates and values to make use of scan
                gates = create_renamed_operators(gates, legit_val_keys)  # type: ignore[index]
                values_map = {
                    gates[ind].param: pname  # type: ignore[index]
                    for pname in legit_val_keys
                    for ind in param_to_gates_indices[pname]
                }
                values = {
                    temp_name: values[values_map[temp_name]] for temp_name in values_map.keys()
                }
                spectral_gap = {
                    temp_name: gates[ind].spectral_gap  # type: ignore[index]
                    for temp_name in values_map.keys()
                    for ind in param_to_gates_indices[values_map[temp_name]]
                }
            else:
                spectral_gap = spectral_gap_from_gates(param_to_gates_indices, legit_val_keys)

    values_array = stack_sp(list(values.values()))
    if values_array.shape[-1] == 1:
        values_array = values_array.squeeze(-1)
    vals_to_dict = values.keys()

    def values_to_dict(x: Array) -> dict[str, Array]:
        return dict(zip(vals_to_dict, x))

    def jvp_caller(carry_grad: Array, pytree_ind_param: dict[str, Array]) -> tuple[Array, Array]:
        shift_vector = pytree_ind_param["shift"]
        spectral_gap = pytree_ind_param["spectral_gap"]
        up_val = values_to_dict(values_array + shift_vector)
        f_up = analytical_gpsr_fwd(state, gates, observable, up_val)
        down_val = values_to_dict(values_array - shift_vector)
        f_down = analytical_gpsr_fwd(state, gates, observable, down_val)
        grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        return carry_grad + grad, grad

    spectral_gap_array = stack_sp(list(spectral_gap.values()))
    tangent_array = stack_sp(list(tangent_dict.values())).reshape((-1, 1))
    pytree_scan = {
        "shift": shift * jnp.eye(len(vals_to_dict), dtype=values_array.dtype),
        "spectral_gap": spectral_gap_array,
    }

    _, grads = jax.lax.scan(jvp_caller, jnp.zeros(fwd.shape), pytree_scan)
    if values_map:
        # need to remap to original parameter names
        grad_dict = dict(zip(values.keys(), grads))
        grads_legit_dict: dict = {name: list() for name in legit_val_keys}
        for temp_name in grad_dict.keys():
            grads_legit_dict[values_map[temp_name]].append(grad_dict[temp_name])
        grads = tuple(stack_sp(grads_legit_dict[name]).sum(axis=0) for name in legit_val_keys)
    jvp = (stack_sp(grads) * tangent_array).sum(axis=0)
    return fwd, jvp.reshape(fwd.shape)
