from __future__ import annotations

from typing import Tuple

from jax import Array, custom_vjp

from horqrux.apply import apply_gate
from horqrux.parametric import Parametric
from horqrux.primitive import Primitive
from horqrux.utils import OperationType, inner


@custom_vjp
def adjoint_expectation(
    state: Array, gates: list[Primitive], observable: list[Primitive], values: dict[str, float]
) -> Array:
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return inner(out_state, projected_state).real


def adjoint_expectation_fwd(
    state: Array, gates: list[Primitive], observable: list[Primitive], values: dict[str, float]
) -> Array:
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return inner(out_state, projected_state).real


def adjoint_expectation_bwd(
    res: Tuple[Array, Array, list[Primitive], dict[str, float]], tangent: Array
) -> tuple:
    """Implementation of Algorithm 1 of https://arxiv.org/abs/2009.02823
    which computes the vector-jacobian product in O(P) time using O(1) state vectors
    where P=number of parameters in the circuit.
    """

    out_state, projected_state, gates, values = res
    grads = {}
    for gate in gates[::-1]:
        out_state = apply_gate(out_state, gate, values, OperationType.DAGGER)
        if isinstance(gate, Parametric):
            mu = apply_gate(out_state, gate, values, OperationType.JACOBIAN)
            grads[gate.param] = tangent * 2 * inner(mu, projected_state).real
        projected_state = apply_gate(projected_state, gate, values, OperationType.DAGGER)
    return (None, None, None, grads)


adjoint_expectation.defvjp(adjoint_expectation_fwd, adjoint_expectation_bwd)
