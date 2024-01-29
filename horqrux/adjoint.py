from __future__ import annotations

from typing import Tuple

from jax import Array, custom_vjp

from horqrux.abstract import Operator, Parametric
from horqrux.apply import apply_gate
from horqrux.utils import OperationType, overlap


def expectation(
    state: Array, gates: list[Operator], observable: list[Operator], values: dict[str, float]
) -> Array:
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return overlap(out_state, projected_state)


@custom_vjp
def adjoint_expectation(
    state: Array, gates: list[Operator], observable: list[Operator], values: dict[str, float]
) -> Array:
    return expectation(state, gates, observable, values)


def adjoint_expectation_fwd(
    state: Array, gates: list[Operator], observable: list[Operator], values: dict[str, float]
) -> Tuple[Array, Tuple[Array, Array, list[Operator], dict[str, float]]]:
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return overlap(out_state, projected_state), (out_state, projected_state, gates, values)


def adjoint_expectation_bwd(
    res: Tuple[Array, Array, list[Operator], dict[str, float]], tangent: Array
) -> tuple:
    out_state, projected_state, gates, values = res
    grads = {}
    for gate in gates[::-1]:
        out_state = apply_gate(out_state, gate, values, OperationType.DAGGER)
        if isinstance(gate, Parametric):
            mu = apply_gate(out_state, gate, values, OperationType.JACOBIAN)
            grads[gate.param] = tangent * 2 * overlap(mu, projected_state)
        projected_state = apply_gate(projected_state, gate, values, OperationType.DAGGER)
    return (None, None, None, grads)


adjoint_expectation.defvjp(adjoint_expectation_fwd, adjoint_expectation_bwd)
