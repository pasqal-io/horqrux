from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from horqrux import expectation, random_state
from horqrux.parametric import RX
from horqrux.primitive import Z

N_QUBITS = 2
SHOTS_ATOL = 0.01
N_SHOTS = 100_000


def test_shots() -> None:
    observables = [Z(0), Z(1)]
    state = random_state(N_QUBITS)
    x = jnp.pi * 0.123
    y = jnp.pi * 0.456

    @functools.partial(jax.jit, static_argnums=2)
    def expect(x, y, method):
        values = {"theta": x}
        ops = [RX("theta", 0), RX(0.2, 0), RX(y, 1), RX("theta", 1)]
        if method == "shots":
            return expectation(state, ops, observables, values, "gpsr", "shots", n_shots=N_SHOTS)
        return expectation(state, ops, observables, values, "ad")

    exp_exact = expect(x, y, "exact")
    exp_shots = expect(x, y, "shots")

    assert jnp.allclose(exp_exact, exp_shots, atol=SHOTS_ATOL)

    d_expect = jax.jit(
        jax.grad(lambda x, y, z: expect(x, y, z).sum(), argnums=[0, 1]), static_argnums=2
    )

    grad_backprop = jnp.stack(d_expect(x, y, "exact"))
    grad_shots = jnp.stack(d_expect(x, y, "shots"))

    assert jnp.allclose(grad_backprop, grad_shots, atol=SHOTS_ATOL)
