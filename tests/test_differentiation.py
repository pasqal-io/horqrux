from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import strategies as st
from absl.testing import parameterized
from hypothesis import given, settings

from horqrux import expectation, random_state, run
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.primitive import Z
from horqrux.utils import density_mat

GPSR_ATOL = 0.05
SHOTS_ATOL = 0.01
N_SHOTS = 100_000


class DifferentiationTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(True, False)
    @given(circuit=st.restricted_circuits())
    @settings(deadline=None)
    def test_shots(self, same_name: bool, circuit: QuantumCircuit) -> None:
        param_names = circuit.param_names
        if same_name:
            ind_change_pname = 0
            for i_op, op in enumerate(circuit.operations):
                if hasattr(op, "param") and op.param == param_names[0]:
                    ind_change_pname = i_op
            ops = circuit.operations
            ops[ind_change_pname].param = param_names[-1]
            circuit = QuantumCircuit(circuit.n_qubits, ops)
            param_names.pop(0)
        x = jax.random.uniform(jax.random.key(0), (len(param_names),))

        def values_to_dict(x):
            return {param_names[i]: x[i] for i in range(len(param_names))}

        observables = (
            [Observable([Z(0)]), Observable([Z(1)])]
            if circuit.n_qubits > 1
            else [
                Observable([Z(0)]),
            ]
        )
        state = random_state(circuit.n_qubits)

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

        assert jnp.allclose(grad_gpsr, grad_shots, atol=SHOTS_ATOL)

        @self.variant
        def dd_exact(x):
            return jax.hessian(lambda x: exact(x).sum())(x)

        @self.variant
        def dd_gpsr(x):
            return jax.hessian(lambda x: exact_gpsr(x).sum())(x)

        assert jnp.allclose(dd_exact(x), dd_gpsr(x), atol=GPSR_ATOL)
