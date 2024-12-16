from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from jax import Array
from jax.tree_util import register_pytree_node_class

from .utils import (
    ErrorProbabilities,
    StrEnum,
)
from .utils_noise import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
)


class NoiseType(StrEnum):
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
class NoiseInstance:
    type: NoiseType
    error_probability: ErrorProbabilities

    def __iter__(self) -> Iterable:
        return iter((self.kraus, self.error_probability))

    def tree_flatten(
        self,
    ) -> tuple[tuple, tuple[NoiseType, ErrorProbabilities]]:
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


NoiseProtocol = tuple[NoiseInstance, ...]
