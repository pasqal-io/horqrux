from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

from horqrux import expectation, random_state, run
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.parametric import RX
from horqrux.primitives.primitive import Z
from horqrux.utils.conversion import to_sparse
from horqrux.utils.operator_utils import density_mat
from tests.utils import verify_arrays

N_QUBITS = 2
GPSR_ATOL = 0.0001
SHOTS_ATOL = 0.01
N_SHOTS = 100_000


class GPSRTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(True, False)
    def test_shots(self, same_name: bool) -> None:
        param_name = "theta"
        if same_name:
            x = jax.random.uniform(jax.random.key(0), (1,))
            ops = [RX(param_name, 0), RX(param_name, 1)]

            def values_to_dict(x):
                return {param_name: x}

        else:
            x = jax.random.uniform(jax.random.key(0), (2,))
            param_names = [param_name, param_name + "2"]
            ops = [RX(param_names[0], 0), RX(param_names[1], 1)]

            def values_to_dict(x):
                return {param_names[0]: x[0], param_names[1]: x[1]}

        circuit = QuantumCircuit(2, ops)
        observables = [Observable([Z(0)]), Observable([Z(1)])]
        state = random_state(N_QUBITS)

        @self.variant
        def exact(x):
            values = values_to_dict(x)
            return expectation(state, circuit, observables, values, diff_mode="ad")

        @self.variant
        def exact_dm(x):
            values = values_to_dict(x)
            return expectation(density_mat(state), circuit, observables, values, diff_mode="ad")

        @self.variant
        def exact_gpsr(x):
            values = values_to_dict(x)
            return expectation(state, circuit, observables, values, diff_mode="gpsr")

        @self.variant
        def exact_gpsr_dm(x):
            values = values_to_dict(x)
            return expectation(density_mat(state), circuit, observables, values, diff_mode="gpsr")

        @self.variant
        def shots(x):
            values = values_to_dict(x)
            return expectation(
                state,
                circuit,
                observables,
                values,
                diff_mode="gpsr",
                n_shots=N_SHOTS,
            )

        @self.variant
        def shots_dm(x):
            values = values_to_dict(x)
            return expectation(
                density_mat(state),
                circuit,
                observables,
                values,
                diff_mode="gpsr",
                n_shots=N_SHOTS,
            )

        expected_dm = density_mat(run(circuit, state, values_to_dict(x)))
        output_dm = run(circuit, density_mat(state), values_to_dict(x))
        assert jnp.allclose(expected_dm.array, output_dm.array)

        exp_exact = exact(x)
        exp_exact_dm = exact_dm(x)
        assert jnp.allclose(exp_exact, exp_exact_dm)

        exp_exact_gpsr = exact_gpsr(x)
        exp_exact_gpsr_dm = exact_gpsr_dm(x)
        assert jnp.allclose(exp_exact_gpsr, exp_exact_gpsr_dm)

        assert jnp.allclose(exp_exact, exp_exact_gpsr, atol=GPSR_ATOL)
        assert jnp.allclose(exp_exact_dm, exp_exact_gpsr_dm, atol=GPSR_ATOL)

        exp_shots = shots(x)
        exp_shots_dm = shots_dm(x)

        assert jnp.allclose(exp_exact, exp_shots, atol=SHOTS_ATOL)
        assert jnp.allclose(exp_exact, exp_shots_dm, atol=SHOTS_ATOL)

        d_exact = jax.grad(lambda x: exact(x).sum())
        d_gpsr = jax.grad(lambda x: exact_gpsr(x).sum())
        d_shots = jax.grad(lambda x: shots(x).sum())

        grad_backprop = d_exact(x)
        grad_gpsr = d_gpsr(x)
        assert jnp.allclose(grad_backprop, grad_gpsr, atol=GPSR_ATOL)

        grad_shots = d_shots(x)

        assert jnp.allclose(grad_backprop, grad_shots, atol=SHOTS_ATOL)

        @self.variant
        def dd_exact(x):
            return jax.hessian(lambda x: exact(x).sum())(x)

        @self.variant
        def dd_gpsr(x):
            return jax.hessian(lambda x: exact_gpsr(x).sum())(x)

        assert jnp.allclose(dd_exact(x), dd_gpsr(x), atol=GPSR_ATOL)

    @chex.variants(with_jit=True, without_jit=True)
    def test_sparse_diff(self) -> None:
        ops = [RX("theta", 0)]
        circuit = QuantumCircuit(2, ops)
        observables = [Observable([Z(0)]), Observable([Z(1)])]
        state = random_state(N_QUBITS)
        x = jnp.pi * 0.5

        @self.variant
        def exact(x):
            values = {"theta": x}
            return expectation(state, circuit, observables, values, diff_mode="ad")

        @self.variant
        def exact_gpsr(x):
            values = {"theta": x}
            return expectation(state, circuit, observables, values, diff_mode="gpsr")

        exp_exact = exact(x)
        exp_gpsr = exact_gpsr(x)

        circuit = to_sparse(circuit)
        observables = [to_sparse(obs) for obs in observables]
        state = random_state(N_QUBITS, sparse=True)

        @self.variant
        def exact_sparse(x):
            values = {"theta": x}
            return expectation(state, circuit, observables, values, diff_mode="ad")

        @self.variant
        def exact_gpsr_sparse(x):
            values = {"theta": x}
            return expectation(state, circuit, observables, values, diff_mode="gpsr")

        exp_exact_sparse = exact_sparse(x)
        exp_gpsr_sparse = exact_gpsr_sparse(x)

        verify_arrays(exp_exact, exp_exact_sparse.todense())
        verify_arrays(exp_gpsr, exp_gpsr_sparse.todense())

        # test gradients raise errors for sparse
        with self.assertRaises(NotImplementedError):
            jax.experimental.sparse.grad(lambda x: exact_gpsr_sparse(x).sum())(x)
