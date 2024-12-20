from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.adjoint import ad_expectation, adjoint_expectation
from horqrux.apply import apply_gate
from horqrux.parametric import RX, RY, Parametric
from horqrux.primitive import NOT, Primitive
from horqrux.utils import DiffMode, zero_state


@register_pytree_node_class
@dataclass
class QuantumCircuit:
    """A minimalistic circuit class to store a sequence of gates.

    Attributes:
        n_qubits (int): Number of qubits.
        operations (list[Primitive]): Operations defining the circuit.
        fparams (list[str]): List of parameters that are considered
            non trainable, used for passing fixed input data to a quantum circuit.
                The corresponding operations compose the `feature map`.
    """

    n_qubits: int
    operations: list[Primitive]
    fparams: list[str] = field(default_factory=list)

    def __call__(self, state: Array, values: dict[str, Array]) -> Array:
        if state is None:
            state = zero_state(self.n_qubits)
        return apply_gate(state, self.operations, values)

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


def expectation(
    state: Array,
    gates: list[Primitive],
    observable: list[Primitive],
    values: dict[str, float],
    diff_mode: DiffMode | str = DiffMode.AD,
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    if diff_mode == DiffMode.AD:
        return ad_expectation(state, gates, observable, values)
    else:
        return adjoint_expectation(state, gates, observable, values)


def sample(
    state: Array,
    gates: list[Primitive],
    values: dict[str, float] = dict(),
    n_shots: int = 1000,
) -> Counter:
    if n_shots < 1:
        raise ValueError("You can only call sample with n_shots>0.")

    wf = apply_gate(state, gates, values)
    probs = jnp.abs(jnp.float_power(wf, 2.0)).ravel()
    key = jax.random.PRNGKey(0)
    n_qubits = len(state.shape)
    # JAX handles pseudo random number generation by tracking an explicit state via a random key
    # For more details, see https://jax.readthedocs.io/en/latest/random-numbers.html
    samples = jax.vmap(
        lambda subkey: jax.random.choice(key=subkey, a=jnp.arange(0, 2**n_qubits), p=probs)
    )(jax.random.split(key, n_shots))

    return Counter(
        {
            format(k, "0{}b".format(n_qubits)): count.item()
            for k, count in enumerate(jnp.bincount(samples))
            if count > 0
        }
    )
