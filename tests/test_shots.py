from __future__ import annotations

import jax.numpy as jnp

from horqrux import expectation, random_state
from horqrux.parametric import RX
from horqrux.primitive import Z

N_QUBITS = 4
SHOTS_ATOL = 0.01
N_SHOTS = 10000


def test_shots() -> None:
    ops = [RX("theta", 0)]
    observable = Z(2)
    values = {p: jnp.ones(1).item() for p in ["theta"]}
    state = random_state(N_QUBITS)

    exp_exact = expectation(state, ops, observable, values, "ad")
    exp_shots = expectation(state, ops, observable, values, "gpsr", "shots", n_shots=N_SHOTS)

    assert jnp.isclose(exp_exact, exp_shots, atol=SHOTS_ATOL)
