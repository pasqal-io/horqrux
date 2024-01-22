from __future__ import annotations

import jax.numpy as jnp
import pytest

from horqrux.apply import apply_gate
from horqrux.parametric import Rx, Ry, Rz
from horqrux.primitive import NOT, SWAP, H, X, Y, Z
from horqrux.utils import equivalent_state, prepare_state


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
    state = apply_gate(state, H(target=0))
    state = apply_gate(state, NOT(target=1, control=0))
    assert jnp.allclose(state.flatten(), final_state)


@pytest.mark.parametrize(
    "x",
    [
        ("10", "01", SWAP(target=(0, 1))),
        ("00", "00", SWAP(target=(0, 1))),
        ("001", "100", SWAP(target=(0, 2))),
        ("011", "110", SWAP(target=(0, 2), control=1)),
        ("001", "001", SWAP(target=(0, 2), control=1)),
        ("00101", "01100", SWAP(target=(4, 1), control=2)),
        ("1001001", "1000011", SWAP(target=(5, 3), control=(6, 0))),
    ],
)
def test_swap_gate(x):
    init_state, expected_state, op = x
    state = prepare_state(len(init_state), init_state)
    out_state = apply_gate(state, op)
    assert equivalent_state(out_state, expected_state), "Output states not similar."


def test_single_gates():
    state = prepare_state(7)
    state = apply_gate(state, X(0, 1))
    state = apply_gate(state, Y(1, 2))
    state = apply_gate(state, Z(2, 4))
    state = apply_gate(state, H(3, 5))
    state = apply_gate(state, Rx(1 / 4 * jnp.pi, 4, 1))
    state = apply_gate(state, Ry(1 / 3 * jnp.pi, 5, 2))
    state = apply_gate(state, Rz(1 / 2 * jnp.pi, 6, 0))
    # FIXME
