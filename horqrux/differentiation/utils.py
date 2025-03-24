from __future__ import annotations
from horqrux.primitives import Parametric, Primitive

def is_parametric(gate: Primitive) -> bool:
    return isinstance(gate, Parametric) and isinstance(gate.param, str)