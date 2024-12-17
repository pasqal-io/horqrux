from __future__ import annotations

import jax
import jax.numpy as jnp

from horqrux import expectation, random_state
from horqrux.parametric import RX
from horqrux.primitive import Z
from horqrux.utils import density_mat

N_QUBITS = 2
SHOTS_ATOL = 0.01
N_SHOTS = 100_000


def test_shots() -> None:
    ops = [RX("theta", 0)]
    observables = [Z(0), Z(1)]
    state = random_state(N_QUBITS)
    x = jnp.pi * 0.5

    def exact(x):
        values = {"theta": x}
        return expectation(state, ops, observables, values, diff_mode="ad")

    def exact_dm(x):
        values = {"theta": x}
        return expectation(density_mat(state), ops, observables, values, diff_mode="ad")

    def shots(x):
        values = {"theta": x}
        return expectation(
            state, ops, observables, values, diff_mode="gpsr", forward_mode="shots", n_shots=N_SHOTS
        )

    def shots_dm(x):
        values = {"theta": x}
        return expectation(
            density_mat(state),
            ops,
            observables,
            values,
            diff_mode="gpsr",
            forward_mode="shots",
            n_shots=N_SHOTS,
        )

    exp_exact = exact(x)
    exp_exact_dm = exact_dm(x)
    assert jnp.allclose(exp_exact, exp_exact_dm)

    exp_shots = shots(x)
    exp_shots_dm = shots_dm(x)

    assert jnp.allclose(exp_exact, exp_shots, atol=SHOTS_ATOL)
    assert jnp.allclose(exp_exact, exp_shots_dm, atol=SHOTS_ATOL)

    d_exact = jax.grad(lambda x: exact(x).sum())
    d_shots = jax.grad(lambda x: shots(x).sum())

    grad_backprop = d_exact(x)
    grad_shots = d_shots(x)

    assert jnp.isclose(grad_backprop, grad_shots, atol=SHOTS_ATOL)
