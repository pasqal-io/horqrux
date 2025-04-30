from __future__ import annotations

import jax
from jax import Array, custom_vjp

from horqrux.apply import apply_gates
from horqrux.composite import Observable, OpSequence
from horqrux.primitives.primitive import Primitive
from horqrux.utils.operator_utils import OperationType, inner
from horqrux.utils.sparse_utils import real_sp


def ad_expectation(
    state: Array, circuit: OpSequence, observable: Observable, values: dict[str, float]
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    out_state = apply_gates(state, list(iter(circuit)), values, OperationType.UNITARY)  # type: ignore[type-var]
    projected_state = observable.forward(out_state, values)
    return real_sp(inner(out_state, projected_state))


@custom_vjp
def __adjoint_expectation_single_observable(
    state: Array, circuit: OpSequence, observable: Observable, values: dict[str, float]
) -> Array:
    return ad_expectation(state, circuit, observable, values)


def adjoint_expectation(
    state: Array,
    circuit: OpSequence,
    observable: Observable,
    values: dict[str, float] | dict[str, dict[str, float]],
) -> Array:
    return ad_expectation(state, circuit, observable, values)  # type: ignore[arg-type]


def adjoint_expectation_single_observable_fwd(
    state: Array, circuit: OpSequence, observable: Observable, values: dict[str, float]
) -> tuple[Array, tuple[Array, Array, list[Primitive], dict[str, float]]]:
    out_state = apply_gates(state, list(iter(circuit)), values, OperationType.UNITARY)  # type: ignore[type-var]
    projected_state = observable.forward(out_state, values)
    return inner(out_state, projected_state).real, (
        out_state,
        projected_state,
        list(iter(circuit)),  # type: ignore[type-var]
        values,
    )


def adjoint_expectation_single_observable_bwd(
    res: tuple[Array, Array, list[Primitive], dict[str, float]], tangent: Array
) -> tuple:
    """Implementation of Algorithm 1 of https://arxiv.org/abs/2009.02823
    which computes the vector-jacobian product in O(P) time using O(1) state vectors
    where P=number of parameters in the circuit.
    """

    out_state, projected_state, gates, values = res
    grads = {k: None for k in values.keys()}

    def gate_bwd_apply(i: int, intermediate_states: tuple[Array, Array]) -> tuple[Array, Array]:
        """Apply a backward operation, that is the dagger representation of gate indexed i.
        One can consider this function as the uncompute part of the method.

        Args:
            i (int): Index of gate.
            intermediate_states (tuple[Array, Array]): The output and projected states before uncomputing.

        Returns:
            tuple[Array, Array]: output and projected states after uncomputing gate i.
        """
        out_state, projected_state = intermediate_states
        gate = gates[i]
        out_state = apply_gates(out_state, gate, values, OperationType.DAGGER)
        if gate.is_parametric:
            mu = apply_gates(out_state, gate, values, OperationType.JACOBIAN)
            grad = tangent * 2 * inner(mu, projected_state).real
            if gate.param not in grads:  # type: ignore[attr-defined]
                grads[gate.param] = grad  # type: ignore[attr-defined]
            else:
                grads[gate.param] += grad  # type: ignore[attr-defined]
        projected_state = apply_gates(projected_state, gate, values, OperationType.DAGGER)
        return out_state, projected_state

    jax.lax.fori_loop(len(gates) - 1, 0, gate_bwd_apply, (out_state, projected_state))
    return (None, None, None, grads)


__adjoint_expectation_single_observable.defvjp(
    adjoint_expectation_single_observable_fwd, adjoint_expectation_single_observable_bwd
)
