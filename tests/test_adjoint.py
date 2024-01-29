from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array, grad

from horqrux import random_state
from horqrux.adjoint import adjoint_expectation, expectation
from horqrux.parametric import PHASE, RX, RY, RZ
from horqrux.primitive import NOT, H, I, S, T, X, Y, Z

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


def test_gradcheck() -> None:
    ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
    observable = [Z(0)]
    values = {
        "theta": np.random.uniform(0, 1),
        "epsilon": np.random.uniform(0, 1),
        "phi": np.random.uniform(0, 1),
        "omega": np.random.uniform(0, 1),
    }
    state = random_state(MAX_QUBITS)

    def adjoint_expfn(values) -> Array:
        return adjoint_expectation(state, ops, observable, values)

    def ad_expfn(values) -> Array:
        return expectation(state, ops, observable, values)

    grads_adjoint = grad(adjoint_expfn)(values)
    grad_ad = grad(ad_expfn)(values)
    for param, ad_grad in grad_ad.items():
        assert jnp.isclose(grads_adjoint[param], ad_grad, atol=0.09)
