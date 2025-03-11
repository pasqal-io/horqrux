from __future__ import annotations

import jax
import jax.numpy as jnp

from horqrux import expectation, random_state, run
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.parametric import RX
from horqrux.primitives.primitive import Z
from horqrux.utils import density_mat
from tests.utils import verify_arrays

N_QUBITS = 2
GPSR_ATOL = 0.0001
SHOTS_ATOL = 0.01
N_SHOTS = 100_000


def test_shots() -> None:
    ops = [RX("theta", 0)]
    circuit = QuantumCircuit(2, ops)
    observables = [Observable([Z(0)]), Observable([Z(1)])]
    state = random_state(N_QUBITS)
    x = jnp.pi * 0.5

    def exact(x):
        values = {"theta": x}
        return expectation(state, circuit, observables, values, diff_mode="ad")

    def exact_dm(x):
        values = {"theta": x}
        return expectation(density_mat(state), circuit, observables, values, diff_mode="ad")

    def exact_gpsr(x):
        values = {"theta": x}
        return expectation(state, circuit, observables, values, diff_mode="gpsr")

    def exact_gpsr_dm(x):
        values = {"theta": x}
        return expectation(density_mat(state), circuit, observables, values, diff_mode="gpsr")

    def shots(x):
        values = {"theta": x}
        return expectation(
            state,
            circuit,
            observables,
            values,
            diff_mode="gpsr",
            forward_mode="shots",
            n_shots=N_SHOTS,
        )

    def shots_dm(x):
        values = {"theta": x}
        return expectation(
            density_mat(state),
            circuit,
            observables,
            values,
            diff_mode="gpsr",
            forward_mode="shots",
            n_shots=N_SHOTS,
        )

    expected_dm = density_mat(run(circuit, state, {"theta": x}))
    output_dm = run(circuit, density_mat(state), {"theta": x})
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
    assert jnp.isclose(grad_backprop, grad_gpsr, atol=GPSR_ATOL)

    grad_shots = d_shots(x)

    assert jnp.isclose(grad_backprop, grad_shots, atol=SHOTS_ATOL)


def test_sparse_diff() -> None:
    ops = [RX("theta", 0, sparse=True)]
    circuit = QuantumCircuit(2, ops)
    observables = [Observable([Z(0, sparse=True)]), Observable([Z(1, sparse=True)])]
    state = random_state(N_QUBITS, sparse=True)
    x = jnp.pi * 0.5

    def exact_sparse(x):
        values = {"theta": x}
        return expectation(state, circuit, observables, values, diff_mode="ad")

    exp_exact_sparse = exact_sparse(x)

    ops = [RX("theta", 0, sparse=False)]
    circuit = QuantumCircuit(2, ops)
    observables = [Observable([Z(0, sparse=False)]), Observable([Z(1, sparse=False)])]
    state = random_state(N_QUBITS, sparse=False)

    def exact(x):
        values = {"theta": x}
        return expectation(state, circuit, observables, values, diff_mode="ad")

    exp_exact = exact(x)

    verify_arrays(exp_exact, exp_exact_sparse.todense())
