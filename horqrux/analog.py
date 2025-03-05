from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

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

    def __post_init__(self) -> None:
        super().__post_init__()
        self._param_uuid = str(uuid4())

    def _unitary(self, values: dict[str, Array] = dict()) -> Array:
        time_val = (
            values[self._param_uuid]
            if self._param_uuid in values.keys()
            else values["time_evolution"]
        )
        return expm(values["hamiltonian"] * (-1j * time_val))


def HamiltonianEvolution(
    target: QubitSupport, control: QubitSupport = (None,)
) -> _HamiltonianEvolution:
    return _HamiltonianEvolution("I", target, control)
