from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random
from pulser import Pulse, Register, Sequence
from pulser.devices import Chadoq2
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation.simulation import Simulation
from scipy.spatial.distance import cdist

# from horqrux.phaser.models.rydberg import QUBOHamiltonian
from horqrux.phaser.simulate import simulate
from horqrux.phaser.test_utils import state_norm, state_overlap
from horqrux.phaser.utils import init_state

key = random.PRNGKey(42)


def test_forward_euler_solver_constant_qubo():
    # Compare pulser to phaser for constant

    n_qubits = 5
    f_rabi = 4.5
    f_detune = 1.5

    def pulser_result():
        # Setting up register
        coords = random.normal(key, (n_qubits, 2))
        distances = cdist(coords, coords)
        min_dist = np.min(distances[distances > 0])
        coords = 6 * coords / min_dist
        reg = Register.from_coordinates(coords)

        param_seq = Sequence(reg, Chadoq2)
        param_seq.declare_channel("ch0", "rydberg_global")
        amplitudes = param_seq.declare_variable("amplitudes", size=2)
        detunings = param_seq.declare_variable("detunings", size=2)
        param_seq.add(
            Pulse(
                InterpolatedWaveform(100, amplitudes),
                InterpolatedWaveform(100, detunings),
                0,
            ),
            "ch0",
        )

        # Detuning of constant 1.0, rabi of 2.0
        seq1 = param_seq.build(amplitudes=[f_rabi, f_rabi], detunings=[f_detune, f_detune])
        sim = Simulation(seq1)
        res = sim.run()

        # Stuff we need to set up Phaser the same
        dt = sim.evaluation_times[1] - sim.evaluation_times[0]
        N = sim.evaluation_times.size + 1
        coords = np.stack(list(reg.qubits.values()), axis=0)
        dists = np.maximum(cdist(coords, coords), 1.0)
        C = Chadoq2.interaction_coeff
        U_inter = np.triu(C / dists**6, k=1)

        return res.get_final_state().full(), (dt, N, U_inter)

    pulser_state, (dt, N, U) = pulser_result()

    # Phaser result
    def phaser_result(dt, N, U):
        # We define a laser function
        # This one is constant but always a function
        def laser(laser_params, t):
            return {
                "rabi": jnp.full((n_qubits,), f_rabi),
                "detune": jnp.full((n_qubits,), f_detune),
            }

        # Initializing Hamiltonian
        in_state = init_state(n_qubits)
        hamiltonian = QUBOHamiltonian(n_qubits, U)
        hamiltonian_params = hamiltonian.init(
            key,
            in_state,
            laser(None, 0.0),
        )
        return simulate(hamiltonian, hamiltonian_params, laser, None, N, dt, in_state)

    phaser_state = phaser_result(dt, N, U)
    assert jnp.allclose(state_norm(phaser_state), 1.0, atol=1e-4)
    assert jnp.allclose(state_overlap(phaser_state, pulser_state), 1.0, atol=1e-2)


def test_forward_euler_time_varying_qubo():
    n_qubits = 5

    def pulser_result():
        # Pulser result
        coords = random.normal(key, (n_qubits, 2))
        distances = cdist(coords, coords)
        min_dist = np.min(distances[distances > 0])
        coords = 6 * coords / min_dist

        reg = Register.from_coordinates(coords)
        param_seq = Sequence(reg, Chadoq2)
        param_seq.declare_channel("ch0", "rydberg_global")
        amplitudes = param_seq.declare_variable("amplitudes", size=5)
        detunings = param_seq.declare_variable("detunings", size=4)
        param_seq.add(
            Pulse(
                InterpolatedWaveform(100, amplitudes),
                InterpolatedWaveform(100, detunings),
                0,
            ),
            "ch0",
        )

        # Detuning of constant 1.0, rabi of 2.0
        seq1 = param_seq.build(amplitudes=[5, 15, 5, 10, 15], detunings=[-10, -20, -15, 0])
        sim = Simulation(seq1)
        res = sim.run()

        # Stuff we need to set up Phaser the same
        dt = sim.evaluation_times[1] - sim.evaluation_times[0]
        N = sim.evaluation_times.size + 1
        coords = np.stack(list(reg.qubits.values()), axis=0)
        dists = np.maximum(cdist(coords, coords), 1.0)
        C = Chadoq2.interaction_coeff
        U_inter = np.triu(C / dists**6, k=1)
        f_rabi = jnp.array(sim.samples["Global"]["ground-rydberg"]["amp"])
        f_detune = jnp.array(sim.samples["Global"]["ground-rydberg"]["det"])

        return res.get_final_state().full(), (dt, N, U_inter, f_rabi, f_detune)

    pulser_state, args = pulser_result()

    def phaser_result(dt, N, U, f_rabi, f_detune):
        def laser(laser_params, t):
            f_rabi, f_detune = laser_params
            return {
                "rabi": jnp.full((n_qubits,), f_rabi[t]),
                "detune": jnp.full((n_qubits,), f_detune[t]),
            }

        laser_params = (f_rabi, f_detune)
        # Initializing Hamiltonian
        in_state = init_state(n_qubits)
        hamiltonian = QUBOHamiltonian(n_qubits, U)
        hamiltonian_params = hamiltonian.init(
            key,
            in_state,
            laser(laser_params, 0),
        )
        return simulate(
            hamiltonian,
            hamiltonian_params,
            laser,
            laser_params,
            N,
            dt,
            in_state,
            iterate_idx=True,
        )

    phaser_state = phaser_result(*args)
    assert jnp.allclose(state_norm(phaser_state), 1.0, atol=1e-4)
    assert jnp.allclose(state_overlap(phaser_state, pulser_state), 1.0, atol=1e-2)


if __name__ == "__main__":
    test_forward_euler_solver_constant_qubo()
    test_forward_euler_time_varying_qubo()
