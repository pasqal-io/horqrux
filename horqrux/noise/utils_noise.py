from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from horqrux.matrices import OPERATIONS_DICT


def BitFlip(error_probability: float) -> tuple[Array, ...]:
    """
    Initialize the BitFlip gate.

    The bit flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p X \\rho X^{\\dagger}

    Args:
        error_probability (float): The probability of a bit flip error.

    Raises:
        ValueError: If the error_probability value is not a float.

    Returns:
        tuple[Array, ...]: Kraus operators for this protocol.
    """
    if (error_probability > 1.0) or (error_probability < 0.0):
        raise ValueError(f"The 'error_probability' value is incorrect. Got {error_probability}.")
    K0: Array = jnp.sqrt(1.0 - error_probability) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(error_probability) * OPERATIONS_DICT["X"]
    kraus: tuple[Array, ...] = (K0, K1)
    return kraus


def PhaseFlip(error_probability: float) -> tuple[Array, ...]:
    """
    Initialize the PhaseFlip gate

    The phase flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p Z \\rho Z^{\\dagger}

    Args:
        error_probability (float): The probability of phase flip error.

    Raises:
        ValueError: If the error_probability value is not a float.

    Returns:
        tuple[Array, ...]: Kraus operators for this protocol.
    """

    if (error_probability > 1.0) or (error_probability < 0.0):
        raise ValueError("The error_probability value is not a correct probability")
    K0: Array = jnp.sqrt(1.0 - error_probability) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(error_probability) * OPERATIONS_DICT["Z"]
    kraus: tuple[Array, ...] = (K0, K1)
    return kraus


def Depolarizing(error_probability: float) -> tuple[Array, ...]:
    """
    Initialize the Depolarizing gate.

    The depolarizing channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho
            + p/3 X \\rho X^{\\dagger}
            + p/3 Y \\rho Y^{\\dagger}
            + p/3 Z \\rho Z^{\\dagger}

    Args:
        error_probability (float): The probability of phase flip error.

    Raises:
        ValueError: If the error_probability value is not a float.

    Returns:
        tuple[Array, ...]: Kraus operators for this protocol.
    """

    if (error_probability > 1.0) or (error_probability < 0.0):
        raise ValueError(f"The 'error_probability' value is incorrect. Got {error_probability}.")
    K0: Array = jnp.sqrt(1.0 - error_probability) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(error_probability / 3.0) * OPERATIONS_DICT["X"]
    K2: Array = jnp.sqrt(error_probability / 3.0) * OPERATIONS_DICT["Y"]
    K3: Array = jnp.sqrt(error_probability / 3.0) * OPERATIONS_DICT["Z"]
    kraus: tuple[Array, ...] = (K0, K1, K2, K3)
    return kraus


def PauliChannel(error_probability: tuple[float, ...]) -> tuple[Array, ...]:
    """
    Initialize the PauliChannel gate.

    The pauli channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-px-py-pz) \\rho
            + px X \\rho X^{\\dagger}
            + py Y \\rho Y^{\\dagger}
            + pz Z \\rho Z^{\\dagger}

    Args:
        error_probability (tuple[float, ...] | float): tuple containing probabilities
            of X, Y, and Z errors.

    Raises:
        ValueError: If the probabilities values do not sum up to 1.

    Returns:
        tuple[Array, ...]: Kraus operators for this protocol.
    """

    sum_prob = sum(error_probability)
    if sum_prob > 1.0:
        raise ValueError("The sum of probabilities can't be greater than 1.0")
    if any([probability > 1.0 or probability < 0.0 for probability in error_probability]):
        raise ValueError(f"The 'error_probability' values are incorrect. Got {error_probability}.")
    px, py, pz = (
        error_probability[0],
        error_probability[1],
        error_probability[2],
    )

    K0: Array = jnp.sqrt(1.0 - (px + py + pz)) * OPERATIONS_DICT["I"]
    K1: Array = jnp.sqrt(px) * OPERATIONS_DICT["X"]
    K2: Array = jnp.sqrt(py) * OPERATIONS_DICT["Y"]
    K3: Array = jnp.sqrt(pz) * OPERATIONS_DICT["Z"]
    kraus: tuple[Array, ...] = (K0, K1, K2, K3)
    return kraus


def AmplitudeDamping(error_probability: float) -> tuple[Array, ...]:
    """
    Initialize the AmplitudeDamping gate.

    The amplitude damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}

    with:

    .. code-block:: python

        K0 = [[1, 0], [0, sqrt(1 - error_probability)]]
        K1 = [[0, sqrt(error_probability)], [0, 0]]

    Args:
        error_probability (float): The damping rate, indicating the probability of amplitude loss.

    Raises:
        ValueError: If the damping rate is not a correct probability.

    Returns:
        tuple[Array, ...]: Kraus operators for this protocol.
    """

    if (error_probability > 1.0) or (error_probability < 0.0):
        raise ValueError(f"The 'error_probability' value is incorrect. Got {error_probability}.")
    K0: Array = jnp.array([[1, 0], [0, jnp.sqrt(1 - error_probability)]], dtype=jnp.complex128)
    K1: Array = jnp.array([[0, jnp.sqrt(error_probability)], [0, 0]], dtype=jnp.complex128)
    kraus: tuple[Array, ...] = (K0, K1)
    return kraus


def PhaseDamping(error_probability: float) -> tuple[Array, ...]:
    """
    Initialize the PhaseDamping gate.

    The phase damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}

     with:

    .. code-block:: python

        K0 = [[1, 0], [0, sqrt(1 - error_probability)]]
        K1 = [[0, 0], [0, sqrt(error_probability)]]

    Args:
        error_probability (float): The damping rate, indicating the probability of phase damping.

    Raises:
        ValueError: If the damping rate is not a correct probability.

    Returns:
        tuple[Array, ...]: Kraus operators for this protocol.
    """

    if (error_probability > 1.0) or (error_probability < 0.0):
        raise ValueError(f"The 'error_probability' value is incorrect. Got {error_probability}.")
    K0: Array = jnp.array([[1, 0], [0, jnp.sqrt(1 - error_probability)]], dtype=jnp.complex128)
    K1: Array = jnp.array([[0, 0], [0, jnp.sqrt(error_probability)]], dtype=jnp.complex128)
    kraus: tuple[Array, ...] = (K0, K1)
    return kraus


def GeneralizedAmplitudeDamping(error_probability: tuple[float, ...]) -> tuple[Array, ...]:
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
        error_probability (tuple[float, ...] | float): The first float must be the probability
            of amplitude damping error, and the second float is the damping rate, indicating
            the probability of generalized amplitude damping.

    Raises:
        ValueError: If the damping rate is not a correct probability.

    Returns:
        tuple[Array, ...]: Kraus operators for this protocol.
    """

    probability = error_probability[0]
    rate = error_probability[1]
    if (probability > 1.0) or (probability < 0.0):
        raise ValueError(
            f"The first value of 'error_probability' value is incorrect. Got {probability}."
        )
    if (rate > 1.0) or (rate < 0.0):
        raise ValueError(f"The second value of 'error_probability' value is incorrect. Got {rate}.")

    K0: Array = jnp.sqrt(probability) * jnp.array(
        [[1, 0], [0, jnp.sqrt(1 - rate)]], dtype=jnp.complex128
    )
    K1: Array = jnp.sqrt(probability) * jnp.array(
        [[0, jnp.sqrt(rate)], [0, 0]], dtype=jnp.complex128
    )
    K2: Array = jnp.sqrt(1.0 - probability) * jnp.array(
        [[jnp.sqrt(1.0 - rate), 0], [0, 1]], dtype=jnp.complex128
    )
    K3: Array = jnp.sqrt(1.0 - probability) * jnp.array(
        [[0, 0], [jnp.sqrt(rate), 0]], dtype=jnp.complex128
    )
    kraus: tuple[Array, ...] = (K0, K1, K2, K3)
    return kraus
