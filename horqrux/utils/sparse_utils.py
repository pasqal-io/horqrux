from __future__ import annotations

import jax.numpy as jnp
from jax.experimental.sparse import sparsify

real_sp = sparsify(jnp.real)
stack_sp = sparsify(jnp.stack)
kron_sp = sparsify(jnp.kron)
eigvalsh_sp = sparsify(jnp.linalg.eigvalsh)
