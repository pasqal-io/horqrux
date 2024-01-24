from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from horqrux.apply import apply_gate, apply_operator
from horqrux.parametric import PHASE, RX, RY, RZ
from horqrux.primitive import NOT, SWAP, H, I, S, T, X, Y, Z
from horqrux.utils import equivalent_state, product_state, random_state

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
def test_primitive(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn(target)
    orig_state = random_state(MAX_QUBITS)
    state = apply_gate(orig_state, gate)
    assert jnp.allclose(
        apply_operator(state, gate.dagger(), gate.target[0], gate.control[0]), orig_state
    )


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
def test_controlled_primitive(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    control = np.random.randint(0, MAX_QUBITS)
    while control == target:
        control = np.random.randint(1, MAX_QUBITS)
    gate = gate_fn(target, control)
    orig_state = random_state(MAX_QUBITS)
    state = apply_gate(orig_state, gate)
    assert jnp.allclose(
        apply_operator(state, gate.dagger(), gate.target[0], gate.control[0]), orig_state
    )


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
def test_parametric(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn("theta", target)
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS)
    state = apply_gate(orig_state, gate, values)
    assert jnp.allclose(
        apply_operator(state, gate.dagger(values), gate.target[0], gate.control[0]), orig_state
    )


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
def test_controlled_parametric(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    control = np.random.randint(0, MAX_QUBITS)
    while control == target:
        control = np.random.randint(1, MAX_QUBITS)
    gate = gate_fn("theta", target, control)
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS)
    state = apply_gate(orig_state, gate, values)
    assert jnp.allclose(
        apply_operator(state, gate.dagger(values), gate.target[0], gate.control[0]), orig_state
    )


@pytest.mark.parametrize(
    ["bitstring", "expected_state"],
    [
        ("00", 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, 1])),
        ("01", 1 / jnp.sqrt(2) * jnp.array([0, 1, 1, 0])),
        ("11", 1 / jnp.sqrt(2) * jnp.array([0, 1, -1, 0])),
        ("10", 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, -1])),
    ],
)
def test_bell_states(bitstring: str, expected_state: Array):
    state = product_state(bitstring)
    state = apply_gate(state, H(target=0))
    state = apply_gate(state, NOT(target=1, control=0))
    assert jnp.allclose(state.flatten(), expected_state)


@pytest.mark.parametrize(
    "inputs",
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
def test_swap_gate(inputs: tuple[str, str, Array]) -> None:
    bitstring, expected_bitstring, op = inputs
    state = product_state(bitstring)
    out_state = apply_gate(state, op)
    assert equivalent_state(out_state, product_state(expected_bitstring))
