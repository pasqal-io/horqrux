from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from horqrux.apply import apply_gates, apply_operator
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, SWAP, H, I, S, T, X, Y, Z
from horqrux.utils.conversion import to_sparse
from horqrux.utils.operator_utils import OperationType, density_mat, product_state, random_state
from tests.utils import verify_arrays

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
@pytest.mark.parametrize("sparse", [False, True])
def test_primitive(gate_fn: Callable, sparse: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn(target, sparse=sparse)
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
@pytest.mark.parametrize("sparse", [False, True])
def test_controlled_primitive(gate_fn: Callable, sparse: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    control = np.random.randint(0, MAX_QUBITS)
    while control == target:
        control = np.random.randint(1, MAX_QUBITS)
    gate = gate_fn(target, control, sparse=sparse)
    orig_state = random_state(MAX_QUBITS, sparse)
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


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
@pytest.mark.parametrize("sparse", [False, True])
def test_parametric(gate_fn: Callable, sparse: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn("theta", target, sparse=sparse)
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS, sparse)
    state = apply_gates(orig_state, gate, values)
    assert verify_arrays(
        apply_operator(state, gate.dagger(values), gate.target[0], gate.control[0]), orig_state
    )

    # test density matrix is similar to pure state
    if not sparse:
        dm = apply_operator(
            density_mat(orig_state),
            gate._unitary(values),
            gate.target[0],
            gate.control[0],
        )
        assert verify_arrays(dm.array, density_mat(state).array)


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
@pytest.mark.parametrize("sparse", [False, True])
def test_controlled_parametric(gate_fn: Callable, sparse: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    control = np.random.randint(0, MAX_QUBITS)
    while control == target:
        control = np.random.randint(1, MAX_QUBITS)
    gate = gate_fn("theta", target, control, sparse=sparse)
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS, sparse=sparse)
    state = apply_gates(orig_state, gate, values)
    assert verify_arrays(
        apply_operator(state, gate.dagger(values), gate.target[0], gate.control[0]), orig_state
    )

    # test density matrix is similar to pure state
    if not sparse:
        dm = apply_operator(
            density_mat(orig_state),
            gate._unitary(values),
            gate.target[0],
            gate.control[0],
        )
        assert verify_arrays(dm.array, density_mat(state).array)


@pytest.mark.parametrize(
    ["bitstring", "expected_state"],
    [
        ("00", 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, 1])),
        ("01", 1 / jnp.sqrt(2) * jnp.array([0, 1, 1, 0])),
        ("11", 1 / jnp.sqrt(2) * jnp.array([0, 1, -1, 0])),
        ("10", 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, -1])),
    ],
)
@pytest.mark.parametrize("sparse", [False, True])
def test_bell_states(bitstring: str, expected_state: Array, sparse: bool):
    state = product_state(bitstring, sparse)
    state = apply_gates(state, H(target=0, sparse=sparse))
    state = apply_gates(state, NOT(target=1, control=0, sparse=sparse))
    if sparse:
        verify_arrays(state.data, expected_state)
    else:
        verify_arrays(state.flatten(), expected_state)


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
@pytest.mark.parametrize("sparse", [False, True])
def test_swap_gate(inputs: tuple[str, str, Array], sparse: bool) -> None:
    bitstring, expected_bitstring, op = inputs
    op.sparse = sparse
    state = product_state(bitstring, sparse=sparse)
    out_state = apply_gates(state, op)
    assert verify_arrays(out_state, product_state(expected_bitstring, sparse=sparse))


@pytest.mark.parametrize("sparse", [False, True])
def test_merge_gates(sparse: bool) -> None:
    gates = [
        RX("a", 0, sparse=sparse),
        RZ("b", 1, sparse=sparse),
        RY("c", 0, sparse=sparse),
        NOT(1, 2, sparse=sparse),
        RX("a", 0, 3, sparse=sparse),
        RZ("c", 3, sparse=sparse),
    ]
    values = {
        "a": np.random.uniform(0.1, 2 * np.pi),
        "b": np.random.uniform(0.1, 2 * np.pi),
        "c": np.random.uniform(0.1, 2 * np.pi),
    }
    state_grouped = apply_gates(
        product_state("0000", sparse=sparse),
        gates,
        values,
        OperationType.UNITARY,
        group_gates=True,
        merge_ops=True,
    )
    state = apply_gates(
        product_state("0000", sparse=sparse),
        gates,
        values,
        OperationType.UNITARY,
        group_gates=False,
        merge_ops=False,
    )
    assert verify_arrays(state_grouped, state)


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
@pytest.mark.parametrize("sparse", [False, True])
def test_cnot_product_state(bitstring: str, sparse: bool):
    cnot0 = NOT(target=1, control=0, sparse=sparse)
    state = product_state(bitstring, sparse=sparse)
    state = apply_gates(state, cnot0)
    expected_state = product_state(flip_bit_wrt_control(bitstring, 0, 1), sparse=sparse)
    assert verify_arrays(state, expected_state)

    # reverse control and target
    cnot1 = NOT(target=0, control=1, sparse=sparse)
    state = product_state(bitstring, sparse=sparse)
    state = apply_gates(state, cnot1)
    expected_state = product_state(flip_bit_wrt_control(bitstring, 1, 0), sparse=sparse)
    assert verify_arrays(state, expected_state)


@pytest.mark.parametrize("sparse", [False, True])
def test_cnot_tensor(sparse: bool) -> None:
    cnot0 = NOT(target=1, control=0, sparse=sparse)
    cnot1 = NOT(target=0, control=1, sparse=sparse)

    t0 = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    t0kron = jnp.kron(t0, jnp.eye(2))
    t1 = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    t1kron = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    if sparse:
        t0 = to_sparse(t0)
        t1 = to_sparse(t1)
        t0kron = to_sparse(t0kron)
        t1kron = to_sparse(t1kron)

    assert verify_arrays(cnot0.tensor(), t0)
    assert verify_arrays(cnot1.tensor(), t1)

    assert verify_arrays(cnot0.tensor(full_support=(0, 1, 2)), t0kron)
    assert verify_arrays(cnot1.tensor(full_support=(0, 1, 2)), t1kron)


@pytest.mark.parametrize("sparse", [False, True])
def test_crx_tensor(sparse: bool) -> None:
    crx0 = RX(0.2, target=1, control=0, sparse=sparse)
    crx1 = RX(0.2, target=0, control=1, sparse=sparse)
    t0 = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.9950, -0.0998j], [0, 0, -0.0998j, 0.9950]])
    t1 = jnp.array([[1, 0, 0, 0], [0, 0.9950, 0, -0.0998j], [0, 0, 1, 0], [0, -0.0998j, 0, 0.9950]])
    if sparse:
        t0 = to_sparse(t0)
        t1 = to_sparse(t1)

    assert verify_arrays(
        crx0.tensor(),
        t0,
        atol=1e-3,
    )
    assert verify_arrays(
        crx1.tensor(),
        t1,
        atol=1e-3,
    )
