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
    """A minimalistic circuit class to store a sequence of gates."""

    n_qubits: int
    operations: list[Primitive]
    feature_map_parameters: list[str] = field(default_factory=list)

    def __call__(self, state: Array, values: dict[str, Array]) -> Array:
        if state is None:
            state = zero_state(self.n_qubits)
        return apply_gate(state, self.operations, values)

    @property
    def param_names(self) -> list[str]:
        return [str(op.param) for op in self.operations if isinstance(op, Parametric)]

    @property
    def variational_param_names(self) -> list[str]:
        return [name for name in self.param_names if name not in self.feature_map_parameters]

    @property
    def n_vparams(self) -> int:
        return len(self.param_names) - len(self.feature_map_parameters)

    def tree_flatten(self) -> tuple:
        children = (
            self.operations,
            self.feature_map_parameters,
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
    by a global entangling operation."""
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
