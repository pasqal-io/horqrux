from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from horqrux.analog import HamiltonianEvolution
from horqrux.apply import apply_gate
from horqrux.utils import overlap, product_state, uniform_state

sigmaz = jnp.diag(jnp.array([1.0, -1.0], dtype=jnp.cdouble))
Hbase = jnp.kron(sigmaz, sigmaz)

Hamiltonian = jnp.kron(Hbase, Hbase)


# @pytest.mark.xfail
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
    time_evo = jnp.array([1.0], dtype=jnp.complex128)
    state = apply_gate(
        state, HamiltonianEvolution((0, 1)), {"hamiltonian": Hbase, "time_evolution": time_evo}
    )
    assert jnp.allclose(state.flatten(), expected_state)


# @pytest.mark.xfail
def test_hamevo_single() -> None:
    n_qubits = 4
    t_evo = jnp.pi / 4
    hamevo = HamiltonianEvolution(tuple([i for i in range(n_qubits)]))
    psi = uniform_state(n_qubits)
    psi_star = apply_gate(psi, hamevo, {"hamiltonian": Hamiltonian, "time_evolution": t_evo})
    result = overlap(psi_star, psi)
    assert jnp.isclose(result, 0.5)
