from __future__ import annotations

from jax.config import config

config.update("jax_enable_x64", True)  # you should really really do this


import jax.numpy as jnp
import pytest
from qiskit import Aer, QuantumCircuit

from horqrux.gates import *
from horqrux.ops import apply_gate
from horqrux.utils import prepare_state

backend = Aer.get_backend("statevector_simulator")


@pytest.mark.skip
def test_single_gates():
    # qiskit
    circuit = QuantumCircuit(7)
    circuit.x(0)
    circuit.y(1)
    circuit.z(2)
    circuit.h(3)
    circuit.rx(1 / 4 * jnp.pi, 4)
    circuit.ry(1 / 3 * jnp.pi, 5)
    circuit.ry(1 / 2 * jnp.pi, 6)
    result = backend.run(circuit).result()
    qiskit_state = result.get_statevector(circuit, decimals=6)

    # horqrux
    state = prepare_state(7)
    state = apply_gate(state, X(0))
    state = apply_gate(state, Y(1))
    state = apply_gate(state, Z(2))
    state = apply_gate(state, H(3))
    state = apply_gate(state, Rx(1 / 4 * jnp.pi, 4))
    state = apply_gate(state, Ry(1 / 3 * jnp.pi, 5))
    state = apply_gate(state, Rz(1 / 2 * jnp.pi, 6))

    assert jnp.allclose(jnp.array(qiskit_state), state.flatten(order="F")), "states not the same!"
