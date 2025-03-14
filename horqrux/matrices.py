from __future__ import annotations

import jax.numpy as jnp
from jax import config
from jax.experimental.sparse import BCOO

from ._misc import default_complex_dtype

config.update("jax_enable_x64", True)  # Quantum ML requires higher precision
default_dtype = default_complex_dtype()

_X = jnp.array([[0, 1], [1, 0]], dtype=default_dtype)
_Y = jnp.array([[0, -1j], [1j, 0]], dtype=default_dtype)
_Z = jnp.array([[1, 0], [0, -1]], dtype=default_dtype)
_H = jnp.array([[1, 1], [1, -1]], dtype=default_dtype) * 1 / jnp.sqrt(2)
_S = jnp.array([[1, 0], [0, 1j]], dtype=default_dtype)
_T = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=default_dtype)
_I = jnp.asarray([[1, 0], [0, 1]], dtype=default_dtype)

_SWAP = jnp.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=default_dtype)


_SQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
        [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
        [0, 0, 0, 1],
    ],
    dtype=default_dtype,
)

_ISWAP = jnp.asarray(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=default_dtype
)

_ISQSWAP = jnp.asarray(
    [
        [1, 0, 0, 0],
        [0, 1 / jnp.sqrt(2), 1j / jnp.sqrt(2), 0],
        [0, 1j / jnp.sqrt(2), 1 / jnp.sqrt(2), 0],
        [0, 0, 0, 1],
    ],
    dtype=default_dtype,
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

SPARSE_OPERATIONS_DICT = {k: BCOO.fromdense(mat) for k, mat in OPERATIONS_DICT.items()}
