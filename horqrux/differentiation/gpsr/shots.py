from __future__ import annotations

from functools import partial
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array, random

from horqrux.composite import Observable
from horqrux.differentiation.gpsr.gpsr_utils import (
    alter_gate_sequence,
    create_renamed_operators,
    extract_gate_names,
    initialize_gpsr_ingredients,
    prepare_param_gates_seq,
    spectral_gap_from_gates,
)
from horqrux.primitives import Primitive
from horqrux.shots import finite_shots
from horqrux.utils.operator_utils import State
from horqrux.utils.sparse_utils import stack_sp

jitted_finite_shots = jax.jit(finite_shots, static_argnums=(4,))


def finite_shots_gpsr_backward(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """Compute the gradient using the GPSR algorithm when using finite shots.

    This function is efficient when used in a
    training loop using the jitted version
    of the `finite_shots` function
    (that is the `horqrux.differentiation.gpsr.jitted_finite_shots`).

    There is also a jvp function `finite_shots_fwd`compatible with `jax.grad`
    but this function may be faster when using `jax.jit`.

    Args:
        state (State): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.

    Returns:
        Array: Gradients.
    """
    val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)

    def jvp_component(param_name: str, key: Array) -> Array:
        up_key, down_key = random.split(key)
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = jitted_finite_shots(state, gates, observables, up_val, n_shots, up_key)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = jitted_finite_shots(state, gates, observables, down_val, n_shots, down_key)
        grad = (
            spectral_gap[param_name]
            * (f_up - f_down)
            / (4.0 * jnp.sin(spectral_gap[param_name] * shift / 2.0))
        )
        return grad

    params_with_keys = zip(values.keys(), random.split(key, len(values)))
    jvp_caller = jvp_component
    if not isinstance(gates, Primitive):
        gate_names = extract_gate_names(gates)
        if len(gate_names) > len(val_keys):
            param_to_gates_indices = prepare_param_gates_seq(val_keys, gates)
            # repeated case
            if max(map(len, param_to_gates_indices.values())) > 1:  # type: ignore[arg-type]

                def jvp_component_repeated_param(param_name: str, key: Array) -> Array:
                    shift_gates = param_to_gates_indices[param_name]
                    shift_keys = random.split(key, len(shift_gates))

                    def shift_jvp(ind: int, key: Array) -> Array:
                        up_key, down_key = random.split(key)
                        spectral_gap = gates[ind].spectral_gap  # type: ignore[index]
                        gates_up = alter_gate_sequence(gates, ind, shift)  # type: ignore[arg-type]
                        f_up = jitted_finite_shots(
                            state, gates_up, observables, values, n_shots, up_key
                        )
                        gates_down = alter_gate_sequence(gates, ind, -shift)  # type: ignore[arg-type]
                        f_down = jitted_finite_shots(
                            state, gates_down, observables, values, n_shots, down_key
                        )
                        return (
                            spectral_gap
                            * (f_up - f_down)
                            / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                        )

                    return sum(
                        shift_jvp(shift_ind, key) for shift_ind, key in zip(shift_gates, shift_keys)
                    )

            else:
                spectral_gap = spectral_gap_from_gates(param_to_gates_indices, val_keys)

            jvp_caller = jvp_component_repeated_param
    jvp = jnp.stack([jvp_caller(param, key) for param, key in params_with_keys]).sum(axis=-1)
    return jvp


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
    fwd = finite_shots_fwd(state, gates, observable, values, n_shots, key)

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

    def jvp_caller(carry_grad: Array, pytree_ind_param: dict[str, Array]) -> Array:
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

    spectral_gap_array = stack_sp(list(spectral_gap.values()))
    tangent_array = stack_sp(list(tangent_dict.values())).reshape((-1, 1))
    keys = random.split(key, len(vals_to_dict))
    pytree_scan = {
        "shift": shift * jnp.eye(len(vals_to_dict), dtype=values_array.dtype),
        "spectral_gap": spectral_gap_array,
        "key": keys,
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
