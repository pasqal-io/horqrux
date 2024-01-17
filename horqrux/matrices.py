from __future__ import annotations

import jax.numpy as jnp
from jax import Array, config
from numpy import log2

from horqrux.utils import hilbert_reshape

config.update("jax_enable_x64", True)  # Quantum ML requires higher precision


def _unitary(theta: float) -> Array:
    return jnp.cos(theta / 2) * jnp.eye(2) - 1j * jnp.sin(theta / 2)


def make_controlled(operator: Array, n_control: int) -> Array:
    n_qubits = int(log2(operator.shape[0]))
    _controlled = jnp.eye(2 ** (n_control + n_qubits))
    _controlled = _controlled.at[-(2**n_qubits) :, -(2**n_qubits) :].set(operator)
    return hilbert_reshape(_controlled)


_X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
_NOT = _X
_Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
_Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
_H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) * 1 / jnp.sqrt(2)
_S = jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex128)
_T = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=jnp.complex128)
_I = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.complex128)

_SWAP = jnp.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex128)


_SQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
        [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
        [0, 0, 0, 1],
    ],
    dtype=jnp.complex128,
)

_ISWAP = jnp.asarray(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex128
)

_ISQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 1 / jnp.sqrt(2), 1j / jnp.sqrt(2), 0],
        [0, 1j / jnp.sqrt(2), 1 / jnp.sqrt(2), 0],
        [0, 0, 0, 1],
    ],
    dtype=jnp.complex128,
)
