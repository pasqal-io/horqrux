from __future__ import annotations

from typing import Tuple

from jax import Array, custom_vjp
from jax.numpy import real as jnpreal

from horqrux.apply import apply_gate
from horqrux.parametric import Parametric
from horqrux.primitive import Primitive
from horqrux.utils import OperationType, inner


def expectation(
    state: Array, gates: list[Primitive], observable: list[Primitive], values: dict[str, float]
) -> Array:
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return jnpreal(inner(out_state, projected_state))


@custom_vjp
def adjoint_expectation(
    state: Array, gates: list[Primitive], observable: list[Primitive], values: dict[str, float]
) -> Array:
    return expectation(state, gates, observable, values)


def adjoint_expectation_fwd(
    state: Array, gates: list[Primitive], observable: list[Primitive], values: dict[str, float]
) -> Tuple[Array, Tuple[Array, Array, list[Primitive], dict[str, float]]]:
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return jnpreal(inner(out_state, projected_state)), (out_state, projected_state, gates, values)


def adjoint_expectation_bwd(
    res: Tuple[Array, Array, list[Primitive], dict[str, float]], tangent: Array
) -> tuple:
    out_state, projected_state, gates, values = res
    grads = {}
    for gate in gates[::-1]:
        out_state = apply_gate(out_state, gate, values, OperationType.DAGGER)
        if isinstance(gate, Parametric):
            mu = apply_gate(out_state, gate, values, OperationType.JACOBIAN)
            grads[gate.param] = tangent * 2 * jnpreal(inner(mu, projected_state))
        projected_state = apply_gate(projected_state, gate, values, OperationType.DAGGER)
    return (None, None, None, grads)


adjoint_expectation.defvjp(adjoint_expectation_fwd, adjoint_expectation_bwd)
