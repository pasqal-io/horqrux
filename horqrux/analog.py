from __future__ import annotations

from jax import Array
from jax.scipy.linalg import expm

from .abstract import Operator, QubitSupport


def HamiltonianEvolution(
    target_idx: QubitSupport,
    control_idx: QubitSupport,
    hamiltonian: Array,
    time_evolution: Array,
) -> Operator:
    """
    A slim wrapper function which evolves a 'hamiltonian'
    given a 'time_evolution' parameter and applies it to 'state' psi by doing: matrixexp(-iHt)|psi>
    """
    return Operator(expm(hamiltonian * (-1j * time_evolution)), target_idx, control_idx)
