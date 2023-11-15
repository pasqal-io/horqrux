from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax.config import config

from horqrux.gates import H, Rx, Ry, Rz, X, Y, Z
from horqrux.ops import apply_gate
from horqrux.utils import prepare_state

config.update("jax_enable_x64", True)  # you should really really do this


@pytest.mark.skip
def test_single_gates():
    # horqrux
    state = prepare_state(7)
    state = apply_gate(state, X(0))
    state = apply_gate(state, Y(1))
    state = apply_gate(state, Z(2))
    state = apply_gate(state, H(3))
    state = apply_gate(state, Rx(1 / 4 * jnp.pi, 4))
    state = apply_gate(state, Ry(1 / 3 * jnp.pi, 5))
    state = apply_gate(state, Rz(1 / 2 * jnp.pi, 6))
    # FIXME
