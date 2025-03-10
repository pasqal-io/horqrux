from __future__ import annotations

from functools import singledispatch
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO


@singledispatch
def verify_arrays(a: Any, b: Any, atol: float = 1e-08) -> bool:
    raise NotImplementedError("verify_arrays is not implemented")


@verify_arrays.register
def _(a: Array, b: Array, atol: float = 1e-08) -> bool:
    return jnp.allclose(a, b, atol)


@verify_arrays.register
def _(a: BCOO, b: BCOO, atol: float = 1e-08) -> bool:
    return jnp.allclose(a.todense(), b.todense(), atol)
