from __future__ import annotations

import jax.numpy as jnp
import pytest
from horqrux.gates import *
from horqrux.ops import apply_gate
from horqrux.utils import prepare_state
from jax.config import config
from qiskit import Aer, QuantumCircuit

config.update("jax_enable_x64", True)  # you should really really do this
backend = Aer.get_backend("statevector_simulator")


@pytest.mark.skip
def test_simple_circuit():
    # Qiskit result
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    result = backend.run(circuit).result()
    qiskit_state = result.get_statevector(circuit, decimals=6)

    # Horqrux
    state = prepare_state(3, "000")
    state = apply_gate(state, H(0))
    state = apply_gate(state, NOT(1, 0))
    horqrux_state = apply_gate(state, NOT(2, 0))

    assert jnp.allclose(jnp.array(qiskit_state), horqrux_state.flatten(order="F"))


@pytest.mark.skip
def test_slightly_more_complicated_circuit():
    # Qiskit result
    circuit = QuantumCircuit(4)
    # First layer
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    # Second layer
    circuit.rx(1 / 4 * jnp.pi, 0)
    circuit.ry(1 / 3 * jnp.pi, 1)
    circuit.rz(1 / 2 * jnp.pi, 2)

    # Third layer
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    # Last layer
    circuit.x(2)
    circuit.y(1)
    circuit.z(0)

    # Result
    result = backend.run(circuit).result()
    qiskit_state = result.get_statevector(circuit, decimals=6)

    # Horqrux
    state = prepare_state(4, "0000")

    # First layer
    state = apply_gate(state, [H(0), NOT(1, 0), NOT(2, 0)])
    state = apply_gate(state, [Rx(1 / 4 * jnp.pi, 0), Ry(1 / 3 * jnp.pi, 1), Rz(1 / 2 * jnp.pi, 2)])
    state = apply_gate(state, [H(0), NOT(1, 0), NOT(2, 0)])
    state = apply_gate(state, [X(2), Y(1), Z(0)])

    assert jnp.allclose(jnp.array(qiskit_state), state.flatten(order="F"))
