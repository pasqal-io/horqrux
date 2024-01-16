from __future__ import annotations

from jax import config

from .analog import HamiltonianEvolution
from .parametric import Rx, Ry, Rz
from .primitive import NOT, SWAP, H, I, S, T, X, Y, Z
from .utils import prepare_state

config.update("jax_enable_x64", True)  # Quantum ML requires higher precision
