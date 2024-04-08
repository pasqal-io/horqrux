from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import add
from uuid import uuid4

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array, jit, value_and_grad, vmap
from numpy.random import uniform

from horqrux import NOT, RX, RY, Z, apply_gate, zero_state
from horqrux.abstract import Primitive
from horqrux.utils import inner

LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
N_POINTS = 150


def ansatz_w_params(n_qubits: int, n_layers: int) -> tuple[list, list]:
    all_ops = []
    param_names = []
    rots_fns = [RX, RY, RX]
    for _ in range(n_layers):
        for i in range(n_qubits):
            ops = [
                fn(str(uuid4()), qubit)
                for fn, qubit in zip(rots_fns, [i for _ in range(len(rots_fns))])
            ]
            param_names += [op.param for op in ops]
            ops += [NOT((i + 1) % n_qubits, i % n_qubits) for i in range(n_qubits)]
            all_ops += ops

    return all_ops, param_names


@dataclass
class TotalMagnetization:
    n_qubits: int

    def __post_init__(self) -> None:
        self.paulis = [Z(i) for i in range(self.n_qubits)]

    def forward(self, state: Array, values: dict) -> Array:
        return reduce(add, [apply_gate(state, pauli, values) for pauli in self.paulis])

    def __call__(self, state: Array, values: dict) -> Array:
        return self.forward(state, values)


@dataclass
class Circuit:
    n_qubits: int
    n_layers: int

    def __post_init__(self) -> None:
        self.feature_map: list[Primitive] = [RX("x", i) for i in range(self.n_qubits // 2)] + [
            RX("y", i) for i in range(self.n_qubits // 2, self.n_qubits)
        ]
        self.ansatz, self.param_names = ansatz_w_params(self.n_qubits, self.n_layers)
        self.observable = TotalMagnetization(self.n_qubits)

    # @partial(vmap, in_axes=(None, None, 0, 0))
    def forward(self, param_values: Array, x: Array, y: Array) -> Array:
        state = zero_state(self.n_qubits)
        param_dict = {name: val for name, val in zip(self.param_names, param_values)}
        out_state = apply_gate(
            state, self.feature_map + self.ansatz, {**param_dict, **{"x": x, "y": y}}
        )
        projected_state = self.observable(state, param_dict)
        return jnp.real(inner(out_state, projected_state))

    def __call__(self, param_values: Array, x: Array, y: Array) -> Array:
        return self.forward(param_values, x, y)

    @property
    def n_vparams(self) -> int:
        return len(self.param_names)


circ = Circuit(N_QUBITS, DEPTH)
# Create random initial values for the parameters
key = jax.random.PRNGKey(42)
params = jax.random.uniform(key, shape=(circ.n_vparams,))

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)


def exp_fn(param_values: Array, x: Array, y: Array) -> Array:
    return circ(param_values, x, y)


def loss_fn(param_values: Array, x: Array, y: Array) -> Array:
    def pde_loss(x: float, y: float) -> Array:
        l_b, r_b, t_b, b_b = list(
            map(
                lambda inputs: exp_fn(param_values, *inputs),
                [
                    [jnp.zeros((1, 1)), y],  # u(0,y)=0
                    [jnp.ones((1, 1)), y],  # u(L,y)=0
                    [x, jnp.ones((1, 1))],  # u(x,H)=0
                    [x, jnp.zeros((1, 1))],  # u(x,0)=f(x)
                ],
            )
        )
        b_b -= jnp.sin(jnp.pi * x)
        hessian = jax.hessian(lambda inputs: exp_fn(params, inputs[0], inputs[1]))(
            jnp.concatenate(
                [
                    x.reshape(
                        1,
                    ),
                    y.reshape(
                        1,
                    ),
                ]
            )
        )
        interior = hessian[0][0] + hessian[1][1]  # uxx+uyy=0
        return reduce(add, list(map(lambda t: jnp.power(t, 2), [l_b, r_b, t_b, b_b, interior])))

    return jnp.mean(vmap(pde_loss, in_axes=(0, 0))(x, y))


def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# collocation points sampling and training
def sample_points(n_in: int, n_p: int) -> Array:
    return uniform(0, 1.0, (n_in, n_p))


@jit
def train_step(i: int, inputs: tuple) -> tuple:
    params, opt_state = inputs
    x, y = sample_points(2, N_POINTS)
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    params, opt_state = optimize_step(params, opt_state, grads)
    return params, opt_state


params, opt_state = jax.lax.fori_loop(0, 1000, train_step, (params, opt_state))
# compare the solution to known ground truth
single_domain = jnp.linspace(0, 1, num=N_POINTS)
domain = jnp.array(list(product(single_domain, single_domain)))
# analytical solution
analytic_sol = (
    (np.exp(-np.pi * domain[:, 0]) * np.sin(np.pi * domain[:, 1])).reshape(N_POINTS, N_POINTS).T
)
# DQC solution

dqc_sol = vmap(lambda domain: exp_fn(params, domain[0], domain[1]), in_axes=(0,))(domain).reshape(
    N_POINTS, N_POINTS
)
# # plot results
fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].imshow(analytic_sol, cmap="turbo")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Analytical solution u(x,y)")
ax[1].imshow(dqc_sol, cmap="turbo")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("DQC solution u(x,y)")
