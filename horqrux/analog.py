from __future__ import annotations

from jax import Array
from jax.scipy.linalg import expm

from .abstract import Primitive, QubitSupport


class _HamiltonianEvolution(Primitive):
    """
    A slim wrapper class which evolves a 'hamiltonian'
    given a 'time_evolution' parameter and applies it to 'state' psi by doing: matrixexp(-iHt)|psi>
    """

    generator_name: str
    target: QubitSupport
    control: QubitSupport

    def unitary(self, values: dict[str, Array] = dict()) -> Array:
        return expm(values["hamiltonian"] * (-1j * values["time_evolution"]))


def HamiltonianEvolution(
    target: QubitSupport, control: QubitSupport = (None,)
) -> _HamiltonianEvolution:
    return _HamiltonianEvolution("I", target, control)
