from __future__ import annotations

from jax.config import config

config.update("jax_enable_x64", True)  # you should really really do this

import time

import jax
import jax.numpy as jnp
import pytest
from jax import random

from horqrux.circuits import DQC
from horqrux.utils import prepare_state

key = random.PRNGKey(42)


@pytest.mark.parametrize(["n_qubits", "n_var_layers", "n_batch"], [(6, 5, 100), (12, 12, 50)])
def test_qcl(n_qubits, n_var_layers, n_batch):
    # Setting up params and inputs
    # 2d in, 3d out
    x = jnp.linspace(0.0, 0.9, n_batch)[:, None]
    X = jnp.concatenate([x, x], axis=1)
    y = jnp.concatenate([x, x**2, x**3], axis=1)

    params = random.uniform(key, (n_var_layers, 3, n_qubits))
    state = prepare_state(n_qubits)
    forward = jax.vmap(lambda params, x: DQC(state, params, x, y.shape[-1]), in_axes=(None, 0))

    def loss_fn(params, x, y):
        y_pred = forward(params, x)
        return jnp.mean((y_pred - y) ** 2)

    f = jax.value_and_grad(loss_fn, argnums=(0,))
    f_jit = jax.jit(f)

    # Fist run compiles so this is slow
    start = time.time()
    print(f(params, X, y)[0].block_until_ready())
    stop = time.time()
    print(f"Without compilation: {stop - start}")

    # After compiling it goes brrrrrrrr.
    jitted_result = f_jit(params, X, y)[0].block_until_ready()
    print(jitted_result)
    start = time.time()
    f_jit(params, X, y)[0].block_until_ready()
    stop = time.time()
    print(f"With compilation: {stop - start}")
