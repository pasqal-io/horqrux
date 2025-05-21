from __future__ import annotations

from functools import partial
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array, random

from horqrux.composite import Observable
from horqrux.differentiation.gpsr.gpsr_utils import (
    initialize_gpsr_ingredients,
)
from horqrux.primitives import Primitive
from horqrux.shots import finite_shots
from horqrux.utils.operator_utils import State


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
    """Jvp version for finite_shots_fwd.

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

    _, spectral_gap, shift = initialize_gpsr_ingredients(values)

    def jvp_component(param_name: str, key: Array) -> Array:
        up_key, down_key = random.split(key)
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = finite_shots_fwd(state, gates, observable, up_val, n_shots, up_key)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = finite_shots_fwd(state, gates, observable, down_val, n_shots, down_key)
        grad = (
            spectral_gap[param_name]
            * (f_up - f_down)
            / (4.0 * jnp.sin(spectral_gap[param_name] * shift / 2.0))
        )
        return grad

    params_with_keys = zip(values.keys(), random.split(key, len(values)))
    fwd = finite_shots_fwd(state, gates, observable, values, n_shots, key)
    jvp = sum(jvp_component(param, key) * tangent_dict[param] for param, key in params_with_keys)
    return fwd, jvp.reshape(fwd.shape)
