from __future__ import annotations

from dataclasses import dataclass

from jax import Array
from jax.scipy.linalg import expm

from .ops import apply_operator
from .types import State


@dataclass
class HamiltonianEvolution:
    """
    A class which performs the Hamiltonian Evolution operation
    Wrapper class which stores the qubits indices on which
    """

    qubits: tuple
    n_qubits: int

    def forward(self, hamiltonian: Array, time_evolution: Array, state: Array) -> State:
        # return expm(-iHt)|psi>
        return apply_operator(state, expm(-1j * hamiltonian * time_evolution), self.qubits, (None,))
