from __future__ import annotations

from jax.config import config

config.update("jax_enable_x64", True)  # you should really really do this


import pytest

from horqrux.gates import SWAP
from horqrux.ops import apply_gate
from horqrux.test_utils import equivalent_state
from horqrux.utils import prepare_state


@pytest.mark.parametrize(
    "x",
    [
        ("10", "01", SWAP(target_idx=(0, 1))),
        ("00", "00", SWAP(target_idx=(0, 1))),
        ("001", "100", SWAP(target_idx=(0, 2))),
        ("011", "110", SWAP(target_idx=(0, 2), control_idx=1)),
        ("001", "001", SWAP(target_idx=(0, 2), control_idx=1)),
        ("00101", "01100", SWAP(target_idx=(4, 1), control_idx=2)),
        ("1001001", "1000011", SWAP(target_idx=(5, 3), control_idx=(6, 0))),
    ],
)
def test_swap_gate(x):
    init_state, expected_state, op = x
    state = prepare_state(len(init_state), init_state)
    out_state = apply_gate(state, op)
    assert equivalent_state(out_state, expected_state), "Output states not similar."
