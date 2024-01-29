from __future__ import annotations

from functools import reduce
from typing import Tuple

from jax import Array, custom_vjp

from horqrux.abstract import Operator, Parametric
from horqrux.apply import apply_operator
from horqrux.utils import overlap


def expectation(
    state: Array, operators: list[Operator], observable: list[Operator], values: dict[str, Array]
) -> Array:
    def forward(state: Array, op: Operator) -> Array:
        return apply_operator(state, op.unitary(values), op.target[0], op.control[0])

    out_state = reduce(forward, operators, state)
    projected_state = reduce(forward, observable, out_state)
    return overlap(out_state, projected_state)


@custom_vjp
def adjoint_expectation(
    state: Array, operators: list[Operator], observable: list[Operator], values: dict[str, Array]
) -> Array:
    return expectation(state, operators, observable, values)


def expectation_fwd(
    state: Array, operators: list[Operator], observable: list[Operator], values: dict[str, Array]
) -> Tuple[Array, Tuple[Array, Array, list[Operator], dict[str, Array]]]:
    def forward(state: Array, op: Operator) -> Array:
        return apply_operator(state, op.unitary(values), op.target[0], op.control[0])

    out_state = reduce(forward, operators, state)
    projected_state = reduce(forward, observable, out_state)
    return overlap(out_state, projected_state), (out_state, projected_state, operators, values)


def expectation_bwd(
    res: Tuple[Array, Array, list[Operator], dict[str, Array]], tangent: Array
) -> tuple:
    out_state, projected_state, operators, values = res
    grads = {}
    for op in operators[::-1]:
        out_state = apply_operator(out_state, op.dagger(values), op.target[0], op.control[0])
        if isinstance(op, Parametric):
            mu = apply_operator(out_state, op.jacobian(values), op.target[0], op.control[0])
            grads[op.param] = tangent * 2 * overlap(mu, projected_state)
        projected_state = apply_operator(
            projected_state, op.dagger(values), op.target[0], op.control[0]
        )
    return (None, None, None, grads)


adjoint_expectation.defvjp(expectation_fwd, expectation_bwd)
