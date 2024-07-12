from __future__ import annotations

import numpy as np

from horqrux import expectation, random_state
from horqrux.parametric import PHASE, RX, RY, RZ
from horqrux.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.shots import finite_shots

MAX_QUBITS = 2
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)


def test_gradcheck() -> None:
    ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
    observable = Z(0)
    values = {
        "theta": np.random.uniform(0, 1),
        "epsilon": np.random.uniform(0, 1),
        "phi": np.random.uniform(0, 1),
        "omega": np.random.uniform(0, 1),
    }
    state = random_state(MAX_QUBITS)
    exp_exact = expectation(state, ops, [observable], values, "ad")
    exp_shots = finite_shots(state, ops, observable, values)
