from __future__ import annotations

import jax.numpy as jnp

_X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
_Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
_Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
_H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) * 1 / jnp.sqrt(2)
_S = jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex64)
_T = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=jnp.complex64)
_I = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.complex64)

_SWAP = jnp.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex64)


_SQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
        [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
        [0, 0, 0, 1],
    ],
    dtype=jnp.complex64,
)

_ISWAP = jnp.asarray(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex64
)

_ISQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 1 / jnp.sqrt(2), 1j / jnp.sqrt(2), 0],
        [0, 1j / jnp.sqrt(2), 1 / jnp.sqrt(2), 0],
        [0, 0, 0, 1],
    ],
    dtype=jnp.complex64,
)

OPERATIONS_DICT = {
    "X": _X,
    "Y": _Y,
    "Z": _Z,
    "H": _H,
    "S": _S,
    "T": _T,
    "I": _I,
    "SWAP": _SWAP,
}
