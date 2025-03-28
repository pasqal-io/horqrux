from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from jax.tree_util import register_pytree_node_class

from horqrux.composite.sequence import OpSequence
from horqrux.primitives.parametric import RX, RY, Parametric
from horqrux.primitives.primitive import NOT, Primitive


@register_pytree_node_class
@dataclass
class QuantumCircuit(OpSequence):
    """A minimalistic circuit class to store a sequence of gates.

    Attributes:
        n_qubits (int): Number of qubits.
        operations (list[Primitive]): Operations defining the circuit.
        fparams (list[str]): List of parameters that are considered
            non trainable, used for passing fixed input data to a quantum circuit.
                The corresponding operations compose the `feature map`.
    """

    def __init__(
        self, n_qubits: int, operations: Primitive | OpSequence | list, fparams: list[str] = list()
    ):
        super().__init__(list(operations))  # type:ignore[arg-type]
        self.n_qubits = n_qubits
        self.fparams = fparams

    @property
    def param_names(self) -> list[str]:
        """List of parameters of the circuit.
            Composed of variational and feature map parameters.

        Returns:
            list[str]: Names of parameters.
        """
        return [str(op.param) for op in self.operations if isinstance(op, Parametric)]

    @property
    def vparams(self) -> list[str]:
        """List of variational parameters of the circuit.

        Returns:
            list[str]: Names of variational parameters.
        """
        return [name for name in self.param_names if name not in self.fparams]

    @property
    def n_vparams(self) -> int:
        """Number of variational parameters.

        Returns:
            int: Number of variational parameters.
        """
        return len(self.param_names) - len(self.fparams)

    def tree_flatten(self) -> tuple:
        children = (
            self.operations,
            self.fparams,
        )
        aux_data = (self.n_qubits,)
        return (aux_data, children)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*aux_data, *children)


def hea(
    n_qubits: int,
    n_layers: int,
    rot_fns: list[Callable] = [RX, RY, RX],
    variational_param_prefix: str = "v_",
) -> list[Primitive]:
    """Hardware-efficient ansatz; A helper function to generate a sequence of rotations followed
    by a global entangling operation.

    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Number of layers
        rot_fns (list[Callable], optional): A list of rotations applied on one qubit.
            Defaults to [RX, RY, RX].
        variational_param_prefix (str, optional): Prefix for the name of variational parameters.
            Defaults to "v_". Names suffix are randomly generated strings with uuid4.

    Returns:
        list[Primitive]: List of gates composing the ansatz.
    """
    gates = []
    param_names = []
    for _ in range(n_layers):
        for i in range(n_qubits):
            ops = [
                fn(variational_param_prefix + str(uuid4()), qubit)
                for fn, qubit in zip(rot_fns, [i for _ in range(len(rot_fns))])
            ]
            param_names += [op.param for op in ops]
            ops += [NOT((i + 1) % n_qubits, i % n_qubits) for i in range(n_qubits)]  # type: ignore[arg-type]
            gates += ops

    return gates
