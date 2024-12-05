from __future__ import annotations

import jax
import jax.numpy as jnp
from jax._src.typing import DType


def default_complex_dtype() -> DType:
    if jax.config.jax_enable_x64:
        return jnp.complex128
    else:
        return jnp.complex64
