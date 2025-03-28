from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.matrices import OPERATIONS_DICT, SPARSE_OPERATIONS_DICT
from horqrux.noise import NoiseProtocol
from horqrux.utils.operator_utils import (
    ControlQubits,
    QubitSupport,
    State,
    TargetQubits,
    _dagger,
    controlled,
    expand_operator,
    is_controlled,
    none_like,
    zero_state,
)


@register_pytree_node_class
@dataclass
class Primitive:
    """Primitive gate class which stores information about generators target and control qubits
    of a particular quantum operator."""

    generator_name: str
    target: QubitSupport
    control: QubitSupport
    noise: NoiseProtocol = None
    sparse: bool = False

    @staticmethod
    def parse_idx(
        idx: tuple,
    ) -> tuple:
        if isinstance(idx, (int, np.int64)):
            return ((idx,),)
        elif isinstance(idx, tuple):
            return (idx,)
        else:
            return (idx.astype(int),)

    def __post_init__(self) -> None:
        self.target = Primitive.parse_idx(self.target)
        if self.control is None:
            self.control = none_like(self.target)
        else:
            self.control = Primitive.parse_idx(self.control)

    def __call__(self, state: State | None = None, values: dict[str, Array] = dict()) -> State:
        from horqrux.apply import apply_gates

        if state is None:
            state = zero_state(len(self.qubit_support))
        return apply_gates(state, self, values)

    def __iter__(self) -> Iterable[Primitive]:
        return iter((self,))

    def tree_flatten(
        self,
    ) -> tuple[tuple, tuple[str, TargetQubits, ControlQubits, NoiseProtocol, bool]]:
        children = ()
        aux_data = (self.generator_name, self.target[0], self.control[0], self.noise, self.sparse)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    @property
    def generator(self) -> Array:
        return (
            SPARSE_OPERATIONS_DICT[self.generator_name]
            if self.sparse
            else OPERATIONS_DICT[self.generator_name]
        )

    def _unitary(self, values: dict[str, float] = dict()) -> Array:
        """Obtain the base unitary from `generator_name`.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().

        Returns:
            Array: The base unitary from `generator_name`.
        """
        return self.generator

    def dagger(self, values: dict[str, float] = dict()) -> Array:
        """Obtain the dagger of the base unitary from `generator_name`.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().

        Returns:
            Array: The base unitary daggered from `generator_name`.
        """
        return _dagger(self._unitary(values))

    def tensor(
        self,
        values: dict[str, float] = dict(),
        full_support: tuple[int, ...] | None = None,
    ) -> Array:
        """Obtain the unitary taking into account the qubit support for controlled operations.

        Args:
            values (dict[str, float], optional): Parameter values. Defaults to dict().
            full_support (tuple[int, ...], optional): The qubit support of definition for the unitary.

        Returns:
            Array: Unitary representation taking into account the qubit support.
        """
        base_unitary = self._unitary(values)
        if is_controlled(self.control):
            base_unitary = controlled(base_unitary, self.target, self.control)
        if full_support is None:
            return base_unitary
        else:
            return expand_operator(base_unitary, self.qubit_support, full_support)

    @property
    def name(self) -> str:
        return "C" + self.generator_name if is_controlled(self.control) else self.generator_name

    @property
    def n_qubits(self) -> int:
        n_qubits = len(self.target)
        if self.control[0] is not None:
            n_qubits += len(self.control)
        return n_qubits

    @property
    def qubit_support(self) -> tuple:
        return tuple(
            sorted(
                tuple(
                    set(
                        self.target[0] + self.control[0]
                        if is_controlled(self.control)
                        else self.target[0]
                    )
                )
            )
        )

    @property
    def is_parametric(self) -> bool:
        return False

    def __repr__(self) -> str:
        return self.name + f"(target={self.target}, control={self.control})"


def I(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """Identity / I gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: I(1) represents the instruction to apply I to qubit 1.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("I", target, control, noise, sparse)


def X(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """The definition for the X gate.

    This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: X(1) represents the instruction to apply X to qubit 1.
    Example usage controlled: X(1, 0) represents the instruction to apply CX / CNOT to qubit 1 with controlled qubit 0.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("X", target, control, noise, sparse)


NOT = X


def Y(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """Y gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: Y(1) represents the instruction to apply X to qubit 1.
    Example usage controlled: Y(1, 0) represents the instruction to apply CY to qubit 1 with controlled qubit 0.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("Y", target, control, noise, sparse)


def Z(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """Z gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: Z(1) represents the instruction to apply Z to qubit 1.
    Example usage controlled: Z(1, 0) represents the instruction to apply CZ to qubit 1 with controlled qubit 0.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("Z", target, control, noise, sparse)


def H(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """H/ Hadamard gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: H(1) represents the instruction to apply Hadamard to qubit 1.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("H", target, control, noise, sparse)


def S(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """S gate or constant phase gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: S(1) represents the instruction to apply S to qubit 1.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("S", target, control, noise, sparse)


def T(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """T gate. This function returns an instance of 'Primitive' and does *not* apply the gate.
    By providing tuple of ints to 'control', it turns into a controlled gate.

    Example usage: T(1) represents the instruction to apply Hadamard to qubit 1.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints indicating the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("T", target, control, noise, sparse)


# Multi (target) qubit gates


def SWAP(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    """SWAP gate. By providing a control, it turns into a controlled gate (Fredkin gate),
       use None for no control qubits.

    Example usage: SWAP((0, 1), (None,)) swaps qubits 0 and 1.
    Example usage controlled: SWAP(((0, 1), ), ((2, ))) swaps qubits 0 and 1 with controlled bit 2.

    Args:
        target: tuple of ints describing the qubits to apply to.
        control: Optional tuple of ints or None describing
        the control qubits. Defaults to (None,).
        noise: The noise instance. Defaults to None.
        sparse: True to use sparse arrays.

    Returns:
        A Primitive instance.
    """
    return Primitive("SWAP", target, control, noise, sparse)


def SQRTSWAP(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    return Primitive("SQSWAP", target, control, noise, sparse)


def ISWAP(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    return Primitive("ISWAP", target, control, noise, sparse)


def ISQRTSWAP(
    target: TargetQubits,
    control: ControlQubits = (None,),
    noise: NoiseProtocol = None,
    sparse: bool = False,
) -> Primitive:
    return Primitive("ISQSWAP", target, control, noise, sparse)
