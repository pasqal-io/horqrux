from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from horqrux.api import expectation, run, sample
from horqrux.apply import apply_gates
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
from horqrux.primitives.parametric import PHASE, RX, RY, RZ
from horqrux.primitives.primitive import NOT, H, I, S, T, X, Y, Z
from horqrux.utils.operator_utils import density_mat, product_state, random_state

MAX_QUBITS = 7
PARAMETRIC_GATES = (RX, RY, RZ, PHASE)
PRIMITIVE_GATES = (NOT, H, X, Y, Z, I, S, T)

NOISE_single_prob = (
    DigitalNoiseType.BITFLIP,
    DigitalNoiseType.PHASEFLIP,
    DigitalNoiseType.DEPOLARIZING,
    DigitalNoiseType.AMPLITUDE_DAMPING,
    DigitalNoiseType.PHASE_DAMPING,
)
ALL_NOISES = list(DigitalNoiseType)


def noise_instance(noise_type: DigitalNoiseType, zero_proba: bool = False) -> DigitalNoiseInstance:
    if noise_type in NOISE_single_prob:
        errors = 0.1 if not zero_proba else 0.0
    elif noise_type == DigitalNoiseType.PAULI_CHANNEL:
        errors = (0.4, 0.5, 0.1) if not zero_proba else (0.0,) * 3
    else:
        errors = (0.2, 0.8) if not zero_proba else (0.0,) * 2

    return DigitalNoiseInstance(noise_type, error_probability=errors)


@pytest.mark.parametrize("noise_type", NOISE_single_prob)
def test_error_prob(noise_type: DigitalNoiseType):
    with pytest.raises(ValueError):
        noise = DigitalNoiseInstance(noise_type, error_probability=-0.5).kraus
    with pytest.raises(ValueError):
        noise = DigitalNoiseInstance(noise_type, error_probability=1.1).kraus


def test_error_paulichannel():
    with pytest.raises(ValueError):
        noise = DigitalNoiseInstance(
            DigitalNoiseType.PAULI_CHANNEL, error_probability=(0.4, 0.5, 1.1)
        ).kraus

    for p in range(3):
        probas = [1.0 / 3.0] * 3
        probas[p] = -0.1
        with pytest.raises(ValueError):
            noise = DigitalNoiseInstance(
                DigitalNoiseType.PAULI_CHANNEL, error_probability=probas
            ).kraus

        probas = [0.0] * 3
        probas[p] = 1.1
        with pytest.raises(ValueError):
            noise = DigitalNoiseInstance(
                DigitalNoiseType.PAULI_CHANNEL, error_probability=probas
            ).kraus


def test_error_prob_GeneralizedAmplitudeDamping():
    for p in range(2):
        probas = [1.0 / 2.0] * 2
        probas[p] = -0.1
        with pytest.raises(ValueError):
            noise = DigitalNoiseInstance(
                DigitalNoiseType.GENERALIZED_AMPLITUDE_DAMPING, error_probability=probas
            ).kraus

        probas = [0.0] * 2
        probas[p] = 1.1
        with pytest.raises(ValueError):
            noise = DigitalNoiseInstance(
                DigitalNoiseType.GENERALIZED_AMPLITUDE_DAMPING, error_probability=probas
            ).kraus


@pytest.mark.parametrize("gate_fn", PRIMITIVE_GATES)
@pytest.mark.parametrize("noise_type", ALL_NOISES)
@pytest.mark.parametrize("zero_proba", [False, True])
def test_noisy_primitive(gate_fn: Callable, noise_type: DigitalNoiseType, zero_proba: bool) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    noise = noise_instance(noise_type, zero_proba)

    noisy_gate = gate_fn(target, noise=(noise,))
    assert len(noisy_gate.noise) == 1

    dm_shape_len = 2 * MAX_QUBITS

    orig_state = random_state(MAX_QUBITS)
    output_dm = apply_gates(orig_state, noisy_gate)

    # check output is a density matrix
    assert len(output_dm.array.shape) == dm_shape_len

    orig_dm = density_mat(orig_state)
    assert len(orig_dm.array.shape) == dm_shape_len
    output_dm2 = apply_gates(
        orig_dm,
        noisy_gate,
    )
    assert jnp.allclose(output_dm2.array, output_dm.array)

    perfect_gate = gate_fn(target)
    perfect_output = density_mat(apply_gates(orig_state, perfect_gate))
    if zero_proba:
        assert jnp.allclose(perfect_output.array, output_dm.array)
    else:
        assert not jnp.allclose(perfect_output.array, output_dm.array)


@pytest.mark.parametrize("gate_fn", PARAMETRIC_GATES)
@pytest.mark.parametrize("noise_type", ALL_NOISES)
@pytest.mark.parametrize("zero_proba", [False, True])
def test_noisy_parametric(
    gate_fn: Callable, noise_type: DigitalNoiseType, zero_proba: bool
) -> None:
    target = np.random.randint(0, MAX_QUBITS)
    noise = noise_instance(noise_type, zero_proba)
    noisy_gate = gate_fn("theta", target, noise=(noise,))
    values = {"theta": np.random.uniform(0.1, 2 * np.pi)}
    orig_state = random_state(MAX_QUBITS)

    dm_shape_len = 2 * MAX_QUBITS

    output_dm = apply_gates(orig_state, noisy_gate, values)
    # check output is a density matrix
    assert len(output_dm.array.shape) == dm_shape_len

    orig_dm = density_mat(orig_state)
    assert len(orig_dm.array.shape) == dm_shape_len

    output_dm2 = apply_gates(
        orig_dm,
        noisy_gate,
        values,
    )
    assert jnp.allclose(output_dm2.array, output_dm.array)

    perfect_gate = gate_fn("theta", target)
    perfect_output = density_mat(apply_gates(orig_state, perfect_gate, values))
    if zero_proba:
        assert jnp.allclose(perfect_output.array, output_dm.array)
    else:
        assert not jnp.allclose(perfect_output.array, output_dm.array)


def simple_depolarizing_test() -> None:
    noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.1),)
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
    sampling_output = sample(
        dm_state,
        ops,
    )
    assert "11" in sampling_output.keys()
    assert "01" in sampling_output.keys()

    # test expectation
    exp_dm = expectation(dm_state, ops, [Z(0)], {})
    assert jnp.allclose(exp_dm, jnp.array([-0.86666667], dtype=jnp.float64))

    # test shots expectation
    exp_dm_shots = expectation(dm_state, ops, [Z(0)], {}, n_shots=1000)
    assert jnp.allclose(exp_dm, exp_dm_shots, atol=1e-02)


def test_error_noisy_gate_sparse() -> None:
    noise_type = ALL_NOISES[0]
    noise = noise_instance(noise_type)

    noisy_gate = X(0, noise=(noise,), sparse=True)
    state = product_state("00", sparse=True)

    with pytest.raises(NotImplementedError):
        state_output = apply_gates(state, noisy_gate)
