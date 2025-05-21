from __future__ import annotations

from functools import partial, singledispatch
from operator import is_not
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array, random
from jax.experimental import checkify

from horqrux.apply import apply_gates
from horqrux.composite import Observable
from horqrux.differentiation.ad import _ad_expectation_single_observable
from horqrux.primitives import Parametric, Primitive
from horqrux.utils.operator_utils import DensityMatrix, State, expand_operator, num_qubits
from horqrux.utils.sparse_utils import stack_sp

def to_shift(gate: Parametric, shift_value: float) -> Any:
    """Create the shifted gate for PSR.

    Args:
        gate (Parametric): Gate to shift.
        shift_value (float): Shift value.

    Returns:
        Any: A new Parametric (Any type due to clsmethod)
    """
    children, aux_data = gate.tree_flatten()
    return Parametric.tree_unflatten(aux_data[:-1] + (aux_data[-1] + shift_value,), children)


def prepare_param_gates_seq(
    param_names: tuple[str, ...], gates: Iterable[Primitive]
) -> dict[str, tuple]:
    """Create a dictionary of parameter names and corresponding parametric gates.

    Args:
        param_names (tuple[str, ...]): Parameters.
        gates (Iterable[Primitive]): Sequence of gates.

    Returns:
        dict[str, tuple]: dictionary of parameter names and corresponding parametric gates.
    """
    param_to_gates: dict[str, tuple] = dict.fromkeys(param_names, tuple())
    for i, gate in enumerate(gates):
        if gate.is_parametric and gate.param in param_names:  # type: ignore[attr-defined]
            param_to_gates[gate.param] += (i,)  # type: ignore[attr-defined]
    return param_to_gates


def alter_gate_sequence(gates: Iterable[Any], ind_alter: int, shift_val: float) -> Any:
    """Create a sequence replacing the `ind_alter` gate by its shifted version.

    Args:
        gates (Iterable[Any]): sequence of gates.
        ind_alter (int): Index of gate to shift.
        shift_val (float): Shift value.

    Returns:
        Any: sequence of gates including shift.
    """
    gate_alter = gates[ind_alter]  # type: ignore[index]
    gate_shift = to_shift(gate_alter, shift_val)
    upper = min(ind_alter + 1, len(gates))  # type: ignore[arg-type]
    gates_seq = gates[:ind_alter] + [gate_shift] + gates[upper:]  # type: ignore[index]
    return gates_seq


def initialize_gpsr_ingredients(
    values: dict[str, float]
) -> tuple[tuple[str, ...], dict[str, float], Array]:
    """Initialize the parameter names, spectral_gap, and shift value for GPSR.

    Args:
        values (dict[str, float]): Parameter values.

    Returns:
        tuple[tuple[str, ...], dict[str, float], Array]: parameter names, spectral_gap, and shift value.
    """
    val_keys = tuple(values.keys())
    spectral_gap = dict.fromkeys(val_keys, 2.0)
    shift = jnp.pi / 2
    return val_keys, spectral_gap, shift


def extract_gate_names(gates: Iterable[Primitive]) -> list:
    """Extract gate names when a gate is parametric.

    Args:
        gates (Iterable[Primitive]): List of gates.

    Returns:
        list: list of names.
    """
    gate_names = list(
        map(
            lambda g: g.param if hasattr(g, "param") and isinstance(g.param, str) else None,
            gates,
        )
    )
    gate_names = list(filter(partial(is_not, None), gate_names))
    return gate_names


def spectral_gap_from_gates(
    param_to_gates_indices: dict[str, tuple], param_names: Iterable[str]
) -> dict[str, Array]:
    """Extract spectral gap from each gate.

    Only works when a parameter name is used by only one gate.

    Args:
        param_to_gates_indices (dict[str, tuple]): dictionary mapping
            parameter name to indices of gates.
        param_names (Iterable[str]): Name of parameters.

    Returns:
        dict[str, Array]: Parameter names mapped with the spectral gap.
    """
    return {param: param_to_gates_indices[param][0].spectral_gap for param in param_names}