from __future__ import annotations

from .api import expectation, run, sample
from .apply import apply_gates, apply_operator
from .circuit import QuantumCircuit
from .composite import Add, Observable, Scale
from .primitives.parametric import PHASE, RX, RY, RZ
from .primitives.primitive import NOT, SWAP, H, I, S, T, X, Y, Z
from .utils.operator_utils import (
    DiffMode,
    equivalent_state,
    hilbert_reshape,
    overlap,
    product_state,
    random_state,
    uniform_state,
    zero_state,
)
