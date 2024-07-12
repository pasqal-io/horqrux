from __future__ import annotations

import jax
import jax.numpy as jnp

from horqrux import expectation, random_state
from horqrux.parametric import PHASE, RX, RY, RZ
from horqrux.primitive import NOT, H, I, S, T, X, Y, Z

MAX_QUBITS = 2
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)
SHOTS_GPSR_ATOL = 0.2
N_SHOTS = 1000


def test_gradcheck() -> None:
    ops = [RX("theta", 0)]
    observable = Z(0)
    values = {p: jnp.ones(1).item() for p in ["theta"]}
    state = random_state(MAX_QUBITS)
    exp_exact = expectation(state, ops, [observable], values, "ad")
    exp_shots = expectation(state, ops, observable, values, "gpsr", "shots")

    assert jnp.isclose(exp_exact, exp_shots, atol=SHOTS_GPSR_ATOL)
    grad_gpsr = jax.grad(
        lambda values: expectation(state, ops, observable, values, "gpsr", "shots")
    )(values)
    grad_ad = jax.grad(lambda values: expectation(state, ops, observable, values))(values)
    for param, ad_grad in grad_ad.items():
        assert jnp.isclose(grad_gpsr[param], ad_grad, atol=SHOTS_GPSR_ATOL)
