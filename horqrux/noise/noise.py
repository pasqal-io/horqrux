from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union

from jax import Array
from jax.tree_util import register_pytree_node_class

from horqrux.utils.operator_utils import StrEnum

from .utils_noise import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
)


class DigitalNoiseType(StrEnum):
    BITFLIP = "BitFlip"
    PHASEFLIP = "PhaseFlip"
    DEPOLARIZING = "Depolarizing"
    PAULI_CHANNEL = "PauliChannel"
    AMPLITUDE_DAMPING = "AmplitudeDamping"
    PHASE_DAMPING = "PhaseDamping"
    GENERALIZED_AMPLITUDE_DAMPING = "GeneralizedAmplitudeDamping"


PROTOCOL_TO_KRAUS_FN: dict[str, Callable] = {
    "BitFlip": BitFlip,
    "PhaseFlip": PhaseFlip,
    "Depolarizing": Depolarizing,
    "PauliChannel": PauliChannel,
    "AmplitudeDamping": AmplitudeDamping,
    "PhaseDamping": PhaseDamping,
    "GeneralizedAmplitudeDamping": GeneralizedAmplitudeDamping,
}


@register_pytree_node_class
@dataclass
class DigitalNoiseInstance:
    type: DigitalNoiseType
    error_probability: tuple[float, ...] | float

    def tree_flatten(
        self,
    ) -> tuple[tuple, tuple[DigitalNoiseType, tuple[float, ...] | float]]:
        children = ()
        aux_data = (self.type, self.error_probability)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    @property
    def kraus(self) -> tuple[Array, ...]:
        kraus_fn: Callable[..., tuple[Array, ...]] = PROTOCOL_TO_KRAUS_FN[self.type]
        return kraus_fn(error_probability=self.error_probability)

    def __repr__(self) -> str:
        return self.type + f"(p={self.error_probability})"


NoiseProtocol = Union[tuple[DigitalNoiseInstance, ...], None]
