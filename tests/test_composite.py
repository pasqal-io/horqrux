from __future__ import annotations

import random
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from horqrux.api import expectation
from horqrux.apply import apply_gates
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Add, Observable, OpSequence, Scale
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.utils.operator_utils import density_mat, random_state
from tests.utils import verify_arrays

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
@pytest.mark.parametrize("sparse", [False, True])
def test_scale(gate_fn: Callable, sparse: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    full_support = tuple(range(MAX_QUBITS))
    gate = gate_fn(target, sparse=sparse)
    scale_gate = Scale(gate, 2.0)
    orig_state = random_state(MAX_QUBITS, sparse=sparse)

    state = apply_gates(orig_state, gate)
    scale_state = scale_gate(orig_state)
    assert verify_arrays(jnp.array(2.0) * state, scale_state)
    assert verify_arrays(
        jnp.array(2.0) * gate.tensor(full_support=full_support),
        scale_gate.tensor(full_support=full_support),
    )

    # try with sequences
    scale_gate_seq = Scale(OpSequence([gate]), 2.0)
    scale_state_seq = scale_gate_seq(orig_state)
    assert verify_arrays(jnp.array(2.0) * state, scale_state_seq)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
@pytest.mark.parametrize("sparse", [False, True])
def test_observable_gate(gate_fn: Callable, sparse: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    full_support = tuple(range(MAX_QUBITS))
    gate = gate_fn(target, sparse=sparse)
    obs_gate = Observable([gate])
    orig_state = random_state(MAX_QUBITS, sparse=sparse)

    state = apply_gates(orig_state, gate)
    obs_state = obs_gate.forward(orig_state)
    assert verify_arrays(state, obs_state)
    assert verify_arrays(
        obs_gate.tensor(full_support=full_support), gate.tensor(full_support=full_support)
    )

    # test density matrix
    if not sparse:
        orig_state = density_mat(orig_state)
        state = apply_gates(orig_state, gate)
        obs_state = obs_gate.forward(orig_state)
        assert verify_arrays(state.array, obs_state.array)


@pytest.mark.parametrize("sparse", [False, True])
def test_sequence(sparse: bool) -> None:
    ops = [
        RX("theta", 0, sparse=sparse),
        RY("epsilon", 0, sparse=sparse),
        RX("phi", 0, sparse=sparse),
        NOT(1, 0, sparse=sparse),
        RX("omega", 0, 1, sparse=sparse),
    ]
    values = {
        "theta": np.random.uniform(0, 1),
        "epsilon": np.random.uniform(0, 1),
        "phi": np.random.uniform(0, 1),
        "omega": np.random.uniform(0, 1),
    }

    circuit = OpSequence(ops)
    assert circuit.qubit_support == (0, 1)

    orig_state = random_state(MAX_QUBITS, sparse=sparse)
    sequence_output = circuit(orig_state, values)
    apply_output = apply_gates(orig_state, ops, values)

    assert verify_arrays(sequence_output, apply_output)

    qc = QuantumCircuit(2, ops)
    qc_output = qc(orig_state, values)
    assert verify_arrays(qc_output, sequence_output)

    # test density matrix
    if not sparse:
        orig_state = density_mat(orig_state)
        sequence_output = circuit(orig_state, values)
        apply_output = apply_gates(orig_state, ops, values)
        assert verify_arrays(sequence_output.array, apply_output.array)


@pytest.mark.parametrize("sparse", [False, True])
def test_add(sparse: bool) -> None:
    num_gates = 2
    full_support = tuple(range(MAX_QUBITS))
    orig_state = random_state(MAX_QUBITS, sparse=sparse)
    chosen_gate_ids = np.random.randint(0, len(PRIMITIVE_GATES), (num_gates,))

    chosen_gates = []
    for id in chosen_gate_ids:
        target = random.choice([i for i in range(MAX_QUBITS)])
        chosen_gates.append(PRIMITIVE_GATES[id](target, sparse=sparse))
    add_operator = Add(chosen_gates)

    add_state = add_operator(orig_state)
    assert verify_arrays(
        add_state,
        apply_gates(orig_state, chosen_gates[0]) + apply_gates(orig_state, chosen_gates[1]),
    )
    assert verify_arrays(
        add_operator.tensor(full_support=full_support),
        chosen_gates[0].tensor(full_support=full_support)
        + chosen_gates[1].tensor(full_support=full_support),
    )


def test_sequence_in_sequence() -> None:
    ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
    values = {
        "theta": np.random.uniform(0, 1),
        "epsilon": np.random.uniform(0, 1),
        "phi": np.random.uniform(0, 1),
        "omega": np.random.uniform(0, 1),
    }
    orig_state = random_state(MAX_QUBITS)
    qc = QuantumCircuit(2, ops)
    qc_output = qc(orig_state, values)
    qc2output = qc(qc_output, values)

    qc_in_qc = QuantumCircuit(2, [qc, qc])
    qc_in_qc_output = qc_in_qc(orig_state, values)

    assert jnp.allclose(qc2output, qc_in_qc_output)

    seq = OpSequence([qc, qc])
    seq_output = seq(orig_state, values)
    assert jnp.allclose(qc2output, seq_output)

    # test expectation
    obs = Observable([Z(0)])
    exp_seq = expectation(orig_state, seq, observables=[obs], values=values)

    exp_qc2output = expectation(qc_output, qc, observables=[obs], values=values)
    assert jnp.allclose(exp_qc2output, exp_seq)

    exp_seq_ad = expectation(orig_state, seq, observables=[obs], values=values, diff_mode="adjoint")
    assert jnp.allclose(exp_qc2output, exp_seq_ad)

    exp_seq_gpsr = expectation(orig_state, seq, observables=[obs], values=values, diff_mode="gpsr")
    assert jnp.allclose(exp_qc2output, exp_seq_gpsr)
