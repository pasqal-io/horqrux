from __future__ import annotations

import unittest

import jax
import numpy as np
from jax import Array, value_and_grad

from horqrux import expectation, random_state
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.utils.conversion import to_sparse
from horqrux.utils.operator_utils import DiffMode
from tests.utils import verify_arrays

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


class TestAdjoint(unittest.TestCase):
    def test_gradcheck(self) -> None:
        ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
        circuit = QuantumCircuit(2, ops)
        circuit_sparse = to_sparse(circuit)
        observable = [Observable([Z(0)])]
        observable_sparse = [to_sparse(observable[0])]
        values = {
            "theta": np.random.uniform(0, 1),
            "epsilon": np.random.uniform(0, 1),
            "phi": np.random.uniform(0, 1),
            "omega": np.random.uniform(0, 1),
        }
        state = random_state(MAX_QUBITS)
        state_sparse = random_state(MAX_QUBITS, sparse=True)

        def exp_fn(values: dict, diff_mode: DiffMode = "ad") -> Array:
            return expectation(state, circuit, observable, values, diff_mode).sum()

        def exp_fn_sparse(values: dict, diff_mode: DiffMode = "ad") -> Array:
            return expectation(
                state_sparse, circuit_sparse, observable_sparse, values, diff_mode
            ).sum()

        exp_ad, grad_ad = value_and_grad(exp_fn)(values)
        exp_adjoint, grads_adjoint = value_and_grad(exp_fn)(values, "adjoint")
        assert verify_arrays(exp_ad, exp_adjoint)

        # note grad does not work for sparse operations
        exp_adjoint_sparse = exp_fn_sparse(values, "adjoint")
        with self.assertRaises(NotImplementedError):
            jax.experimental.sparse.grad(exp_fn_sparse)(values, "adjoint")

        assert verify_arrays(exp_adjoint, exp_adjoint_sparse)

        for param, ad_grad in grad_ad.items():
            assert verify_arrays(grads_adjoint[param], ad_grad, atol=1.0e-3)
            # assert verify_arrays(grads_adjoint_sparse[param], grads_adjoint[param])
