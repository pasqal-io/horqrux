from __future__ import annotations

import jax.numpy as jnp

from horqrux.analog import HamiltonianEvolution
from horqrux.apply import apply_gate
from horqrux.utils import overlap, uniform_state

sigmaz = jnp.diag(jnp.array([1.0, -1.0], dtype=jnp.cdouble))
Hbase = jnp.kron(sigmaz, sigmaz)

Hamiltonian = jnp.kron(Hbase, Hbase)


def test_hamevo_single() -> None:
    n_qubits = 4
    t_evo = jnp.pi / 4
    hamevo = HamiltonianEvolution(tuple([i for i in range(n_qubits)]))
    psi = uniform_state(n_qubits)
    psi_star = apply_gate(psi, hamevo, {"hamiltonian": Hamiltonian, "time_evolution": t_evo})
    result = overlap(psi_star, psi)
    assert jnp.isclose(result, 0.5)
