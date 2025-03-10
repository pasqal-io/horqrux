from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from horqrux.apply import apply_gates, apply_operator
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, SWAP, H, I, S, T, X, Y, Z
from horqrux.utils import OperationType, density_mat, equivalent_state, product_state, random_state
from tests.utils import verify_arrays

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
@pytest.mark.parametrize("sparse", [False, True])
def test_primitive(gate_fn: Callable, sparse: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn(target, sparse)
    orig_state = random_state(MAX_QUBITS, sparse)
    assert len(orig_state) == 2
    state = apply_gates(orig_state, gate)
    assert verify_arrays(
        apply_operator(state, gate.dagger(), gate.target[0], gate.control[0]), orig_state
    )

    # test density matrix is similar to pure state
    if not sparse:
        dm = apply_operator(
            density_mat(orig_state),
            gate._unitary(),
            gate.target[0],
            gate.control[0],
        )
        assert verify_arrays(dm.array, density_mat(state).array)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
def test_controlled_primitive(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    control = np.random.randint(0, MAX_QUBITS)
    while control == target:
        control = np.random.randint(1, MAX_QUBITS)
    gate = gate_fn(target, control)
    orig_state = random_state(MAX_QUBITS)
    state = apply_gates(orig_state, gate)
    assert jnp.allclose(
        apply_operator(state, gate.dagger(), gate.target[0], gate.control[0]), orig_state
    )

    # test density matrix is similar to pure state
    dm = apply_operator(
        density_mat(orig_state),
        gate._unitary(),
        gate.target[0],
        gate.control[0],
    )
    assert jnp.allclose(dm.array, density_mat(state).array)


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
def test_parametric(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn("theta", target)
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS)
    state = apply_gates(orig_state, gate, values)
    assert jnp.allclose(
        apply_operator(state, gate.dagger(values), gate.target[0], gate.control[0]), orig_state
    )

    # test density matrix is similar to pure state
    dm = apply_operator(
        density_mat(orig_state),
        gate._unitary(values),
        gate.target[0],
        gate.control[0],
    )
    assert jnp.allclose(dm.array, density_mat(state).array)


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
def test_controlled_parametric(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    control = np.random.randint(0, MAX_QUBITS)
    while control == target:
        control = np.random.randint(1, MAX_QUBITS)
    gate = gate_fn("theta", target, control)
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS)
    state = apply_gates(orig_state, gate, values)
    assert jnp.allclose(
        apply_operator(state, gate.dagger(values), gate.target[0], gate.control[0]), orig_state
    )

    # test density matrix is similar to pure state
    dm = apply_operator(
        density_mat(orig_state),
        gate._unitary(values),
        gate.target[0],
        gate.control[0],
    )
    assert jnp.allclose(dm.array, density_mat(state).array)


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
    state = apply_gates(state, H(target=0))
    state = apply_gates(state, NOT(target=1, control=0))
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
    out_state = apply_gates(state, op)
    assert equivalent_state(out_state, product_state(expected_bitstring))


def test_merge_gates() -> None:
    gates = [RX("a", 0), RZ("b", 1), RY("c", 0), NOT(1, 2), RX("a", 0, 3), RZ("c", 3)]
    values = {
        "a": np.random.uniform(0.1, 2 * np.pi),
        "b": np.random.uniform(0.1, 2 * np.pi),
        "c": np.random.uniform(0.1, 2 * np.pi),
    }
    state_grouped = apply_gates(
        product_state("0000"),
        gates,
        values,
        OperationType.UNITARY,
        group_gates=True,
        merge_ops=True,
    )
    state = apply_gates(
        product_state("0000"),
        gates,
        values,
        OperationType.UNITARY,
        group_gates=False,
        merge_ops=False,
    )
    assert jnp.allclose(state_grouped, state)


def flip_bit_wrt_control(bitstring: str, control: int, target: int) -> str:
    # Convert bitstring to list for easier manipulation
    bits = list(bitstring)

    # Flip the bit at the specified index
    if bits[control] == "1":
        bits[target] = "0" if bits[target] == "1" else "1"

    # Convert back to string
    return "".join(bits)


@pytest.mark.parametrize(
    "bitstring",
    [
        "00",
        "01",
        "11",
        "10",
    ],
)
def test_cnot_product_state(bitstring: str):
    cnot0 = NOT(target=1, control=0)
    state = product_state(bitstring)
    state = apply_gates(state, cnot0)
    expected_state = product_state(flip_bit_wrt_control(bitstring, 0, 1))
    assert jnp.allclose(state, expected_state)

    # reverse control and target
    cnot1 = NOT(target=0, control=1)
    state = product_state(bitstring)
    state = apply_gates(state, cnot1)
    expected_state = product_state(flip_bit_wrt_control(bitstring, 1, 0))
    assert jnp.allclose(state, expected_state)


def test_cnot_tensor() -> None:
    cnot0 = NOT(target=1, control=0)
    cnot1 = NOT(target=0, control=1)
    assert jnp.allclose(
        cnot0.tensor(), jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    )
    assert jnp.allclose(
        cnot1.tensor(), jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    )


def test_crx_tensor() -> None:
    crx0 = RX(0.2, target=1, control=0)
    crx1 = RX(0.2, target=0, control=1)
    assert jnp.allclose(
        crx0.tensor(),
        jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.9950, -0.0998j], [0, 0, -0.0998j, 0.9950]]),
        atol=1e-3,
    )
    assert jnp.allclose(
        crx1.tensor(),
        jnp.array([[1, 0, 0, 0], [0, 0.9950, 0, -0.0998j], [0, 0, 1, 0], [0, -0.0998j, 0, 0.9950]]),
        atol=1e-3,
    )
