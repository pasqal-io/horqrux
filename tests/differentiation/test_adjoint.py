from __future__ import annotations

import chex
import jax
import numpy as np
from absl.testing import parameterized
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


class TestAdjoint(chex.TestCase):
    @chex.variants(with_jit=False, without_jit=True)
    @parameterized.parameters(True, False)
    def test_gradcheck(self, same_name: bool) -> None:
        names = ["theta", "epsilon", "phi", "omega"]
        if same_name:
            while len(set(names)) > 3:
                names = np.random.choice(names, len(names))
        ops = [
            RX(names[0], 0),
            RY(names[1], 0),
            RX(names[2], 0),
            NOT(1, 0),
            RX(names[3], 0, 1),
            NOT(1, 0),
            RY(1.0, 0),
        ]
        circuit = QuantumCircuit(2, ops)
        circuit_sparse = to_sparse(circuit)
        observable = [Observable([Z(0)])]
        observable_sparse = [to_sparse(observable[0])]
        values = {name: np.random.uniform(0, 1) for name in set(names)}
        state = random_state(MAX_QUBITS)
        state_sparse = random_state(MAX_QUBITS, sparse=True)

        @self.variant(static_argnums=(1,))
        def exp_fn(values: dict, diff_mode: DiffMode = "ad") -> Array:
            return expectation(state, circuit, observable, values, diff_mode).sum()

        @self.variant(static_argnums=(1,))
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
