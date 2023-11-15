from __future__ import annotations

import jax.numpy as jnp
import pytest

from horqrux.gates import NOT, H
from horqrux.ops import apply_gate
from horqrux.utils import prepare_state


@pytest.mark.parametrize(
    ["init_state", "final_state"],
    [
        ("00", 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, 1])),
        ("01", 1 / jnp.sqrt(2) * jnp.array([0, 1, 1, 0])),
        ("11", 1 / jnp.sqrt(2) * jnp.array([0, 1, -1, 0])),
        ("10", 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, -1])),
    ],
)
def test_bell_states(init_state, final_state):
    state = prepare_state(len(init_state), init_state)
    state = apply_gate(state, H(target_idx=0))
    state = apply_gate(state, NOT(target_idx=1, control_idx=0))
    assert jnp.allclose(state.flatten(), final_state)
