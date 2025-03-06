from __future__ import annotations

from dataclasses import dataclass

from jax import Array
from jax.scipy.linalg import expm
from jax.tree_util import register_pytree_node_class

from horqrux.primitives.primitive import Primitive, QubitSupport


@register_pytree_node_class
@dataclass
class _HamiltonianEvolution(Primitive):
    """
    A slim wrapper class which evolves a 'hamiltonian'
    given a 'time_evolution' parameter and applies it to 'state' psi by doing: matrixexp(-iHt)|psi>
    """

    generator_name: str
    target: QubitSupport
    control: QubitSupport
    shift: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

    def _unitary(self, values: dict[str, Array] = dict()) -> Array:
        # note: GPSR trick when the same param_name is used in many operations
        time_val = values["time_evolution"] + self.shift
        return expm(values["hamiltonian"] * (-1j * time_val))


def HamiltonianEvolution(
    target: QubitSupport, control: QubitSupport = (None,)
) -> _HamiltonianEvolution:
    return _HamiltonianEvolution("I", target, control)
