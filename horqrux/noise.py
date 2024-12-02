from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from .utils import (
    QubitSupport,
    TargetQubits,
    ErrorProbabilities,
)
from .matrices import OPERATIONS_DICT



@register_pytree_node_class
@dataclass
class Noise:
    """Noise class which stores information on ."""

    kraus: list[Array]
    target: QubitSupport
    error_probability: ErrorProbabilities

    @staticmethod
    def parse_idx(
        idx: Tuple,
    ) -> Tuple:
        if isinstance(idx, (int, np.int64)):
            return ((idx,),)
        elif isinstance(idx, tuple):
            return (idx,)
        else:
            return (idx.astype(int),)

    def __post_init__(self) -> None:
        self.target = Noise.parse_idx(self.target)

    def __iter__(self) -> Iterable:
        return iter((self.kraus, self.target, self.error_probability))

    def tree_flatten(self) -> Tuple[Tuple, Tuple[list[Array], TargetQubits, ErrorProbabilities]]:
        children = ()
        aux_data = (self.kraus, self.target[0], self.error_probability)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

def BitFlip(target: TargetQubits, error_probability: float) -> Noise:
    """
    Initialize the BitFlip gate.

    The bit flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p X \\rho X^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The probability of a bit flip error.

    Raises:
        ValueError: If the error_probability value is not a float.

    Returns:
        Noise: Noise instance for this protocol.
    """
    if error_probability > 1.0 or error_probability < 0.0:
        raise ValueError("The error_probability value is not a correct probability")
    K0: Array = jnp.sqrt(1.0 - error_probability) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(error_probability) * OPERATIONS_DICT["X"]
    kraus_bitflip: list[Array] = [K0, K1]
    return Noise(kraus_bitflip, target, error_probability)

def PhaseFlip(target: TargetQubits, error_probability: float) -> Noise:
    """
    Initialize the PhaseFlip gate

    The phase flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The probability of phase flip error.

    Raises:
        ValueError: If the error_probability value is not a float.

    Returns:
        Noise: Noise instance for this protocol.
    """
    
    if error_probability > 1.0 or error_probability < 0.0:
        raise ValueError("The error_probability value is not a correct probability")
    K0: Array = jnp.sqrt(1.0 - error_probability) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(error_probability) * OPERATIONS_DICT["Z"]
    kraus: list[Array] = [K0, K1]
    return Noise(kraus, target, error_probability)

def Depolarizing(target: TargetQubits, error_probability: float) -> Noise:
    """
    Initialize the Depolarizing gate.

    The depolarizing channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho
            + p/3 X \\rho X^{\\dagger}
            + p/3 Y \\rho Y^{\\dagger}
            + p/3 Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The probability of phase flip error.

    Raises:
        ValueError: If the error_probability value is not a float.

    Returns:
        Noise: Noise instance for this protocol.
    """
    
    if error_probability > 1.0 or error_probability < 0.0:
        raise ValueError("The error_probability value is not a correct probability")
    K0: Array = jnp.sqrt(1.0 - error_probability) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(error_probability / 3.0) * OPERATIONS_DICT["X"]
    K2: Array = jnp.sqrt(error_probability / 3.0) * OPERATIONS_DICT["Y"]
    K3: Array = jnp.sqrt(error_probability / 3.0) * OPERATIONS_DICT["Z"]
    kraus: list[Array] = [K0, K1, K2, K3]
    return Noise(kraus, target, error_probability)

def PauliChannel(target: TargetQubits, error_probability:  tuple[float, ...]) -> Noise:
    """
    Initialize the PauliChannel gate.

    The pauli channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-px-py-pz) \\rho
            + px X \\rho X^{\\dagger}
            + py Y \\rho Y^{\\dagger}
            + pz Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (ErrorProbabilities): Tuple containing probabilities
            of X, Y, and Z errors.

    Raises:
        ValueError: If the probabilities values do not sum up to 1.

    Returns:
        Noise: Noise instance for this protocol.
    """
    
    sum_prob = sum(error_probability)
    if sum_prob > 1.0:
        raise ValueError("The sum of probabilities can't be greater than 1.0")
    for probability in error_probability:
        if probability > 1.0 or probability < 0.0:
            raise ValueError("The probability values are not correct probabilities")
    px, py, pz = (
        error_probability[0],
        error_probability[1],
        error_probability[2],
    )


    K0: Array = jnp.sqrt(1.0 - (px + py + pz)) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(px) * OPERATIONS_DICT["X"]
    K2: Array = jnp.sqrt(py) * OPERATIONS_DICT["Y"]
    K3: Array = jnp.sqrt(pz) * OPERATIONS_DICT["Z"]
    kraus: list[Array] = [K0, K1, K2, K3]
    return Noise(kraus, target, error_probability)

def AmplitudeDamping(target: TargetQubits, error_probability: float) -> Noise:
    """
    Initialize the AmplitudeDamping gate.

    The amplitude damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}

    with:

    .. code-block:: python

        K0 = [[1, 0], [0, sqrt(1 - rate)]]
        K1 = [[0, sqrt(rate)], [0, 0]]
    
    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The damping rate, indicating the probability of amplitude loss.

    Raises:
        ValueError: If the damping rate is not a correct probability.

    Returns:
        Noise: Noise instance for this protocol.
    """

    rate = error_probability
    if rate > 1.0 or rate < 0.0:
        raise ValueError("The damping rate is not a correct probability")
    K0: Array = jnp.array([[1, 0], [0, jnp.sqrt(1 - rate)]], dtype=jnp.complex128) 
    K1: Array = jnp.array([[0, jnp.sqrt(rate)], [0, 0]], dtype=jnp.complex128) 
    kraus: list[Array] = [K0, K1]
    return Noise(kraus, target, error_probability)

def PhaseDamping(target: TargetQubits, error_probability: float) -> Noise:
    """
    Initialize the PhaseDamping gate.

    The phase damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}

     with:

    .. code-block:: python

        K0 = [[1, 0], [0, sqrt(1 - rate)]]
        K1 = [[0, 0], [0, sqrt(rate)]]

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The damping rate, indicating the probability of phase damping.

    Raises:
        ValueError: If the damping rate is not a correct probability.

    Returns:
        Noise: Noise instance for this protocol.
    """

    rate = error_probability
    if rate > 1.0 or rate < 0.0:
        raise ValueError("The damping rate is not a correct probability")
    K0: Array = jnp.array([[1, 0], [0, jnp.sqrt(1 - rate)]], dtype=jnp.complex128) 
    K1: Array = jnp.array([[0, 0], [0, jnp.sqrt(rate)]], dtype=jnp.complex128) 
    kraus: list[Array] = [K0, K1]
    return Noise(kraus, target, error_probability)

def GeneralizedAmplitudeDamping(target: TargetQubits, error_probability: tuple[float, ...]) -> Noise:
    """
    Initialize the GeneralizeAmplitudeDamping gate.

    The generalize amplitude damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}
            + K_2 \\rho K_2^{\\dagger} + K_3 \\rho K_3^{\\dagger}

    with:

    .. code-block:: python

        K0 = sqrt(p) * [[1, 0], [0, sqrt(1 - rate)]]
        K1 = sqrt(p) * [[0, sqrt(rate)], [0, 0]]
        K2 = sqrt(1-p) * [[sqrt(1 - rate), 0], [0, 1]]
        K3 = sqrt(1-p) * [[0, 0], [sqrt(rate), 0]]

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (ErrorProbabilities): The first float must be the probability
            of amplitude damping error, and the second float is the damping rate, indicating
            the probability of generalized amplitude damping.

    Raises:
        ValueError: If the damping rate is not a correct probability.

    Returns:
        Noise: Noise instance for this protocol.
    """

    probability = error_probability[0]
    rate = error_probability[1]
    if probability > 1.0 or probability < 0.0:
        raise ValueError("The probability value is not a correct probability")
    if rate > 1.0 or rate < 0.0:
        raise ValueError("The damping rate is not a correct probability")
    
    K0: Array = jnp.sqrt(probability) * jnp.array([[1, 0], [0, jnp.sqrt(1 - rate)]], dtype=jnp.complex128) 
    K1: Array = jnp.sqrt(probability) * jnp.array([[0, jnp.sqrt(rate)], [0, 0]], dtype=jnp.complex128) 
    K2: Array = jnp.sqrt(1.0-probability) * jnp.array([[jnp.sqrt(1.0-rate), 0], [0, 1]], dtype=jnp.complex128) 
    K3: Array = jnp.sqrt(1.0-probability) * jnp.array([[0, 0], [jnp.sqrt(rate), 0]], dtype=jnp.complex128)
    kraus: list[Array] = [K0, K1, K2, K3]
    return Noise(kraus, target, error_probability)