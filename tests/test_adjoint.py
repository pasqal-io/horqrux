from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array, grad

from horqrux import expectation, random_state
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.utils import DiffMode

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


def test_gradcheck() -> None:
    ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
    circuit = QuantumCircuit(2, ops)
    observable = [Observable([Z(0)])]
    values = {
        "theta": np.random.uniform(0, 1),
        "epsilon": np.random.uniform(0, 1),
        "phi": np.random.uniform(0, 1),
        "omega": np.random.uniform(0, 1),
    }
    state = random_state(MAX_QUBITS)

    def exp_fn(values: dict, diff_mode: DiffMode = "ad") -> Array:
        return expectation(state, circuit, observable, values, diff_mode).item()

    grads_adjoint = grad(exp_fn)(values, "adjoint")
    grad_ad = grad(exp_fn)(values)
    for param, ad_grad in grad_ad.items():
        assert jnp.isclose(grads_adjoint[param], ad_grad, atol=1.0e-3)
