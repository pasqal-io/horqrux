from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from horqrux.apply import apply_gate
from horqrux.noise import NoiseInstance, NoiseType
from horqrux.parametric import PHASE, RX, RY, RZ
from horqrux.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.utils import density_mat, random_state

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)

NOISE_oneproba = (
    NoiseType.BITFLIP,
    NoiseType.PHASEFLIP,
    NoiseType.DEPOLARIZING,
    NoiseType.AMPLITUDE_DAMPING,
    NoiseType.PHASE_DAMPING,
)
ALL_NOISES = list(NoiseType)


def noise_instance(noise_type: NoiseType) -> NoiseInstance:
    if noise_type in NOISE_oneproba:
        errors = 0.1
    elif noise_type == NoiseType.PAULI_CHANNEL:
        errors = (0.4, 0.5, 0.1)
    else:
        errors = (0.2, 0.8)

    return NoiseInstance(noise_type, error_probability=errors)


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
@pytest.mark.parametrize("noise_type", ALL_NOISES)
def test_noisy_primitive(gate_fn: Callable, noise_type: NoiseType) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    noise = noise_instance(noise_type)

    noisy_gate = gate_fn(target, noise=(noise,))
    assert len(noisy_gate.noise) == 1

    dm_shape_len = 2 * MAX_QUBITS

    orig_state = random_state(MAX_QUBITS)
    output_dm = apply_gate(orig_state, noisy_gate)
    # check output is a density matrix
    assert len(output_dm.shape) == dm_shape_len

    orig_dm = density_mat(orig_state)
    assert len(orig_dm.shape) == dm_shape_len
    output_dm2 = apply_gate(orig_dm, noisy_gate, is_state_densitymat=True)
    assert jnp.allclose(output_dm2, output_dm)


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
@pytest.mark.parametrize("noise_type", ALL_NOISES)
def test_noisy_parametric(gate_fn: Callable, noise_type: NoiseType) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    noise = noise_instance(noise_type)
    noisy_gate = gate_fn("theta", target, noise=(noise,))
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS)

    dm_shape_len = 2 * MAX_QUBITS

    output_dm = apply_gate(orig_state, noisy_gate, values)
    # check output is a density matrix
    assert len(output_dm.shape) == dm_shape_len

    orig_dm = density_mat(orig_state)
    assert len(orig_dm.shape) == dm_shape_len

    output_dm2 = apply_gate(orig_dm, noisy_gate, values, is_state_densitymat=True)
    assert jnp.allclose(output_dm2, output_dm)
