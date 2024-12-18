from __future__ import annotations

from jax import Array, custom_vjp

from horqrux.apply import apply_gate
from horqrux.parametric import Parametric
from horqrux.primitive import GateSequence, Primitive
from horqrux.utils import OperationType, inner


def ad_expectation(
    state: Array, gates: list[Primitive], observables: list[Primitive], values: dict[str, float]
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observables, values, OperationType.UNITARY)
    return inner(out_state, projected_state).real


@custom_vjp
def __adjoint_expectation_single_observable(
    state: Array, gates: list[Primitive], observable: Primitive, values: dict[str, float]
) -> Array:
    return ad_expectation(state, gates, [observable], values)


def adjoint_expectation(
    state: Array, gates: GateSequence, observables: list[Primitive], values: dict[str, float]
) -> Array:
    return ad_expectation(state, gates, observables, values)  # type: ignore[arg-type]


def adjoint_expectation_single_observable_fwd(
    state: Array, gates: list[Primitive], observable: list[Primitive], values: dict[str, float]
) -> tuple[Array, tuple[Array, Array, list[Primitive], dict[str, float]]]:
    out_state = apply_gate(state, gates, values, OperationType.UNITARY)
    projected_state = apply_gate(out_state, observable, values, OperationType.UNITARY)
    return inner(out_state, projected_state).real, (out_state, projected_state, gates, values)


def adjoint_expectation_single_observable_bwd(
    res: tuple[Array, Array, list[Primitive], dict[str, float]], tangent: Array
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


__adjoint_expectation_single_observable.defvjp(
    adjoint_expectation_single_observable_fwd, adjoint_expectation_single_observable_bwd
)
