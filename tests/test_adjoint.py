from __future__ import annotations

import numpy as np
from jax import Array, grad

from horqrux import expectation, random_state
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.utils import DiffMode
from tests.utils import verify_arrays

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


def test_gradcheck() -> None:
    ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
    ops_sparse = [
        RX("theta", 0, sparse=True),
        RY("epsilon", 0, sparse=True),
        RX("phi", 0, sparse=True),
        NOT(1, 0, sparse=True),
        RX("omega", 0, 1, sparse=True),
    ]
    circuit = QuantumCircuit(2, ops)
    circuit_sparse = QuantumCircuit(2, ops_sparse)
    observable = [Observable([Z(0)])]
    observable_sparse = [Observable([Z(0, sparse=True)])]
    values = {
        "theta": np.random.uniform(0, 1),
        "epsilon": np.random.uniform(0, 1),
        "phi": np.random.uniform(0, 1),
        "omega": np.random.uniform(0, 1),
    }
    state = random_state(MAX_QUBITS)
    state_sparse = random_state(MAX_QUBITS, sparse=True)

    def exp_fn(values: dict, diff_mode: DiffMode = "ad") -> Array:
        return expectation(state, circuit, observable, values, diff_mode).item()

    def exp_fn_sparse(values: dict, diff_mode: DiffMode = "ad") -> Array:
        return (
            expectation(state_sparse, circuit_sparse, observable_sparse, values, diff_mode)
            .todense()
            .item()
        )

    grad_ad = grad(exp_fn)(values)
    grads_adjoint = grad(exp_fn)(values, "adjoint")
    grads_adjoint_sparse = grad(exp_fn_sparse)(values, "adjoint")

    for param, ad_grad in grad_ad.items():
        assert verify_arrays(grads_adjoint[param], ad_grad, atol=1.0e-3)
        assert verify_arrays(grads_adjoint_sparse[param], grads_adjoint[param])
