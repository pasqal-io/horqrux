from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Union

import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from .utils import none_like

# Type aliases for target and control indices.
TargetIdx = Tuple[Tuple[int, ...], ...]
ControlIdx = Tuple[Union[None, Tuple[int, ...]], ...]

# State is just an array but this clarifies type annotation
State = Array


@register_pytree_node_class
@dataclass
class Gate:
    O: ArrayLike
    target_idx: TargetIdx
    control_idx: ControlIdx

    def __post_init__(self) -> None:
        self.target_idx = self.parse_idx(self.target_idx)
        if self.control_idx is None:
            self.control_idx = none_like(self.target_idx)
        else:
            self.control_idx = self.parse_idx(self.control_idx)

    @staticmethod
    def parse_idx(
        idx: Tuple,
    ) -> Tuple:
        if isinstance(idx, int):
            return ((idx,),)
        elif isinstance(idx, np.int64):  # for some weird reason...
            return ((idx,),)
        elif isinstance(idx, tuple):
            return (idx,)
        else:
            return (idx.astype(int),)

    def __iter__(self) -> Iterable:
        return iter((self.O, self.target_idx, self.control_idx))

    # For Jax vmap etc to work
    def tree_flatten(self) -> Tuple[Tuple[Array], Tuple[TargetIdx, ControlIdx]]:
        children = (self.O,)
        aux_data = (
            self.target_idx,
            self.control_idx,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)
