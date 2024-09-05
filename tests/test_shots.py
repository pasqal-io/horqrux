from __future__ import annotations

import jax
import jax.numpy as jnp

from horqrux import expectation, random_state
from horqrux.parametric import RX
from horqrux.primitive import Z

N_QUBITS = 4
SHOTS_ATOL = 0.01
N_SHOTS = 10000


def test_shots() -> None:
    ops = [RX("theta", 0)]
    observable = Z(0)
    state = random_state(N_QUBITS)
    x = jnp.pi * 0.5

    def exact(x):
        values = {"theta": x}
        return expectation(state, ops, observable, values, "ad")

    def shots(x):
        values = {"theta": x}
        return expectation(state, ops, observable, values, "gpsr", "shots", n_shots=N_SHOTS)

    exp_exact = exact(x)
    exp_shots = exact(x)

    assert jnp.isclose(exp_exact, exp_shots, atol=SHOTS_ATOL)

    d_exact = jax.grad(exact)
    d_shots = jax.grad(shots)

    grad_backprop = d_exact(x)
    grad_shots = d_shots(x)

    assert jnp.isclose(grad_backprop, grad_shots, atol=SHOTS_ATOL)
