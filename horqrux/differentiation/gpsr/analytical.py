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
    extract_gate_names,
    initialize_gpsr_ingredients,
    prepare_param_gates_seq,
    spectral_gap_from_gates,
)
from horqrux.primitives import Primitive
from horqrux.utils.operator_utils import State
from horqrux.utils.sparse_utils import stack_sp


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def no_shots_fwd(
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


@no_shots_fwd.defjvp
def no_shots_fwd_jvp(
    state: Array,
    gates: Union[Primitive, Iterable[Primitive]],
    observable: list[Observable],
    primals: tuple[dict[str, float]],
    tangents: tuple[dict[str, float]],
) -> Array:
    values = primals[0]
    tangent_dict = tangents[0]

    val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)

    def jvp_component(param_name: str) -> Array:
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = no_shots_fwd(state, gates, observable, up_val)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = no_shots_fwd(state, gates, observable, down_val)
        grad = (
            spectral_gap[param_name]
            * (f_up - f_down)
            / (4.0 * jnp.sin(spectral_gap[param_name] * shift / 2.0))
        )
        return grad

    fwd = no_shots_fwd(state, gates, observable, values)
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
                        f_up = no_shots_fwd(state, gates_up, observable, values)
                        gates_down = alter_gate_sequence(gates, ind, -shift)
                        f_down = no_shots_fwd(state, gates_down, observable, values)
                        return (
                            spectral_gap
                            * (f_up - f_down)
                            / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                        )

                    return sum(shift_jvp(shift_ind) for shift_ind in shift_gates)

                jvp_caller = jvp_component_repeated_param
            else:
                spectral_gap = spectral_gap_from_gates(param_to_gates_indices, val_keys)

    jvp = sum(jvp_caller(param) * tangent_dict[param] for param in val_keys)
    return fwd, jvp.reshape(fwd.shape)
