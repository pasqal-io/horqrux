from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from horqrux.api import expectation, run, sample
from horqrux.apply import apply_gate
from horqrux.noise import NoiseInstance, NoiseType
from horqrux.parametric import PHASE, RX, RY, RZ
from horqrux.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.utils import density_mat, product_state, random_state

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

    perfect_gate = gate_fn(target)
    perfect_output = density_mat(apply_gate(orig_state, perfect_gate))
    assert not jnp.allclose(perfect_output, output_dm)


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

    perfect_gate = gate_fn("theta", target)
    perfect_output = density_mat(apply_gate(orig_state, perfect_gate, values))
    assert not jnp.allclose(perfect_output, output_dm)


def simple_depolarizing_test() -> None:
    noise = (NoiseInstance(NoiseType.DEPOLARIZING, 0.1),)
    ops = [X(0, noise=noise), X(1)]
    state = product_state("00")
    state_output = run(ops, state)

    # test run
    assert jnp.allclose(
        state_output,
        jnp.array(
            [
                [
                    [[0.0 - 0.0j, 0.0 - 0.0j], [0.0 - 0.0j, 0.0 - 0.0j]],
                    [[0.0 - 0.0j, 0.06666667 - 0.0j], [0.0 - 0.0j, 0.0 - 0.0j]],
                ],
                [
                    [[0.0 - 0.0j, 0.0 - 0.0j], [0.0 - 0.0j, 0.0 - 0.0j]],
                    [[0.0 - 0.0j, 0.0 - 0.0j], [0.0 - 0.0j, 0.93333333 - 0.0j]],
                ],
            ],
            dtype=jnp.complex128,
        ),
    )

    # test sampling
    dm_state = density_mat(state)
    sampling_output = sample(dm_state, ops, is_state_densitymat=True)
    assert "11" in sampling_output.keys()
    assert "01" in sampling_output.keys()

    # test expectation
    exp_dm = expectation(dm_state, ops, [Z(0)], {})
    assert jnp.allclose(exp_dm, jnp.array([-0.86666667], dtype=jnp.float64))
