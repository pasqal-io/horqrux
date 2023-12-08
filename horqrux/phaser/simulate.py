from __future__ import annotations

from horqrux.phaser.solvers import forward_euler_solve


def simulate(
    hamiltonian, hamiltonian_params, laser, laser_params, N, dt, in_state, **solver_kwargs
):
    def propagate_fn(params, state, t, dt):
        hamiltonian_params, laser_params = params
        return hamiltonian.apply(
            hamiltonian_params,
            state,
            dt,
            laser(laser_params, t),
            method=hamiltonian.evolve,
        )

    return forward_euler_solve(
        in_state, propagate_fn, (hamiltonian_params, laser_params), N=N, dt=dt, **solver_kwargs
    )
