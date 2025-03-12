from __future__ import annotations

from typing import Any, Callable

import hypothesis.strategies as st
from hypothesis.strategies._internal import SearchStrategy

from horqrux.circuit import QuantumCircuit
from horqrux.primitives.parametric import RX, RY, RZ
from horqrux.primitives.primitive import H, I, Primitive, X, Y, Z

digital_gateset = [I, H, X, Y, Z, RX, RY, RZ]
param_gateset = [RX, RY, RZ]

MIN_CIRCUIT_DEPTH = 1
MAX_CIRCUIT_DEPTH = 3
MIN_N_QUBITS = 2
MAX_N_QUBITS = 4

N_QUBITS_STRATEGY: SearchStrategy[int] = st.integers(min_value=MIN_N_QUBITS, max_value=MAX_N_QUBITS)
CIRCUIT_DEPTH_STRATEGY: SearchStrategy[int] = st.integers(
    min_value=MIN_CIRCUIT_DEPTH, max_value=MAX_CIRCUIT_DEPTH
)


# A strategy to generate random blocks.
def rand_circuits(gate_list: list[Primitive]) -> Callable:
    @st.composite
    def blocks(
        # ops_pool: list[AbstractBlock] TO BE ADDED
        draw: Callable[[SearchStrategy[Any]], Any],
        n_qubits: SearchStrategy[int] = st.integers(min_value=1, max_value=4),
        depth: SearchStrategy[int] = st.integers(min_value=1, max_value=8),
    ) -> QuantumCircuit:
        total_qubits = draw(n_qubits)
        gates_list: list = []
        qubit_indices = {0}

        for _ in range(draw(depth)):
            gate = draw(st.sampled_from(gate_list))

            qubit = draw(st.integers(min_value=0, max_value=total_qubits - 1))
            qubit_indices = qubit_indices.union({qubit})

            if total_qubits == 1:
                if gate in param_gateset:
                    gates_list.append(gate(target=qubit, param=st.text()))
                else:
                    gates_list.append(gate(target=qubit))
            else:
                is_controlled = draw(st.booleans())
                if is_controlled:
                    target = draw(
                        st.integers(min_value=0, max_value=total_qubits - 1).filter(
                            lambda x: x != qubit
                        )
                    )
                    qubit_indices = qubit_indices.union({target})
                    if gate in param_gateset:
                        gates_list.append(gate(target=target, control=qubit, param=st.text()))
                    else:
                        gates_list.append(gate(target=target, control=qubit))

        return QuantumCircuit(total_qubits, gates_list)

    return blocks


@st.composite
def restricted_circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = N_QUBITS_STRATEGY,
    depth: SearchStrategy[int] = CIRCUIT_DEPTH_STRATEGY,
) -> QuantumCircuit:
    circuit = draw(rand_circuits(digital_gateset)(n_qubits, depth))
    return circuit
