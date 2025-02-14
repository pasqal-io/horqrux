from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
import random

from horqrux.apply import apply_gate, apply_operator
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.composite import Scale, Add, Sequence, Observable
from horqrux.circuit import QuantumCircuit
from horqrux.utils import random_state, density_mat

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
def test_scale(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn(target)
    scale_gate = Scale(gate, 2.0)
    orig_state = random_state(MAX_QUBITS)

    state = apply_gate(orig_state, gate)
    scale_state = scale_gate(orig_state)
    assert jnp.allclose(
        jnp.array(2.0) * state, scale_state
    )

@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
def test_observable_gate(gate_fn: Callable) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    gate = gate_fn(target)
    obs_gate = Observable([gate])
    orig_state = random_state(MAX_QUBITS)

    state = apply_gate(orig_state, gate)
    obs_state = obs_gate(orig_state)
    assert jnp.allclose(
        state, obs_state
    )

    # test density matrix
    orig_state = density_mat(orig_state)
    state = apply_gate(orig_state, gate)
    obs_state = obs_gate(orig_state)
    assert jnp.allclose(
        state.array, obs_state.array
    )

def test_sequence() -> None:
    ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
    values = {
        "theta": np.random.uniform(0, 1),
        "epsilon": np.random.uniform(0, 1),
        "phi": np.random.uniform(0, 1),
        "omega": np.random.uniform(0, 1),
    }

    circuit = Sequence(ops)
    assert circuit.qubit_support == (0,1)

    orig_state = random_state(MAX_QUBITS)
    sequence_output = circuit(orig_state, values)
    apply_output = apply_gate(orig_state, ops, values)

    assert jnp.allclose(
        sequence_output, apply_output
    )

    qc = QuantumCircuit(2, ops)
    qc_output = qc(orig_state, values)
    assert jnp.allclose(
        qc_output, sequence_output
    )

    # test density matrix
    orig_state = density_mat(orig_state)
    sequence_output = circuit(orig_state, values)
    apply_output = apply_gate(orig_state, ops, values)
    assert jnp.allclose(
        sequence_output.array, apply_output.array
    )



def test_add() -> None:
    num_gates = 2
    orig_state = random_state(MAX_QUBITS)
    chosen_gate_ids = np.random.randint(0, len(PRIMITIVE_GATES), (num_gates,))

    chosen_gates = []
    for id in chosen_gate_ids:
        target = random.choice([i for i in range(MAX_QUBITS)])
        chosen_gates.append(PRIMITIVE_GATES[id](target))
    add_operator = Add(chosen_gates)

    add_state = add_operator(orig_state)
    assert jnp.allclose(
        add_state, apply_gate(orig_state, chosen_gates[0]) + apply_gate(orig_state, chosen_gates[1])
    )