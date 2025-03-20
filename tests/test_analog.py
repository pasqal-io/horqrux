from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, vmap

from horqrux.analog import HamiltonianEvolution
from horqrux.apply import apply_gates
from horqrux.utils.operator_utils import is_normalized, overlap, random_state, uniform_state

sigmaz = jnp.diag(jnp.array([1.0, -1.0], dtype=jnp.cdouble))
Hbase = jnp.kron(sigmaz, sigmaz)

Hamiltonian = jnp.kron(Hbase, Hbase)


def test_hamevo_single() -> None:
    n_qubits = 4
    t_evo = jnp.pi / 4
    hamevo = HamiltonianEvolution(tuple([i for i in range(n_qubits)]))
    psi = uniform_state(n_qubits)
    psi_star = apply_gates(psi, hamevo, {"hamiltonian": Hamiltonian, "time_evolution": t_evo})
    result = overlap(psi_star, psi)
    assert jnp.isclose(result, 0.5)


def Hamiltonian_general(n_qubits: int = 2, batch_size: int = 1) -> jnp.array:
    H_batch = jnp.zeros((batch_size, 2**n_qubits, 2**n_qubits), dtype=jnp.cdouble)
    for i in range(batch_size):
        H_0 = np.random.uniform(0.0, 1.0, (2**n_qubits, 2**n_qubits)).astype(np.cdouble)
        H = H_0 + jnp.conj(H_0.transpose(0, 1))
        H_batch.at[i, :, :].set(H)
    return H_batch


@pytest.mark.parametrize(
    "n_qubits, batch_size",
    [
        (2, 1),
    ],
)
def test_hamevo_general(n_qubits: int, batch_size: int) -> None:
    H = Hamiltonian_general(n_qubits, batch_size)
    t_evo = np.random.uniform(0, 1, (batch_size, 1))
    hamevo = HamiltonianEvolution(tuple([i for i in range(n_qubits)]))
    psi = random_state(n_qubits)
    psi_star = jit(
        vmap(apply_gates, in_axes=(None, None, {"hamiltonian": 0, "time_evolution": 0}))
    )(psi, hamevo, {"hamiltonian": H, "time_evolution": t_evo})
    assert jnp.all(vmap(is_normalized, in_axes=(0,))(psi_star))
