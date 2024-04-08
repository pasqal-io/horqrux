# Welcome to horqrux

**horqrux** is a state vector simulator designed for quantum machine learning written in [JAX](https://jax.readthedocs.io/).

## Setup

To install `horqrux` , you can go into any virtual environment of your
choice and install it normally with `pip`:

```bash
pip install horqrux
```

## Digital operations

`horqrux` implements a large selection of both primitive and parametric single to n-qubit, digital quantum gates.

Let's have a look at primitive gates first.

```python exec="on" source="material-block"
from horqrux import X, random_state, apply_gate

state = random_state(2)
new_state = apply_gate(state, X(0))
```

We can also make any gate controlled, in the case of X, we have to pass the target qubit first!

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux import X, product_state, equivalent_state, apply_gate

n_qubits = 2
state = product_state('11')
control = 0
target = 1
# This is equivalent to performing CNOT(0,1)
new_state= apply_gate(state, X(target,control))
assert jnp.allclose(new_state, product_state('10'))
```

When applying parametric gates, we can either pass a numeric value or a parameter name for the parameter as the first argument.

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux import RX, random_state, apply_gate

target_qubit = 1
state = random_state(target_qubit+1)
param_value = 1 / 4 * jnp.pi
new_state = apply_gate(state, RX(param_value, target_qubit))
# Parametric horqrux gates also accept parameter names in the form of strings.
# Simply pass a dictionary of parameter names and values to the 'apply_gate' function
new_state = apply_gate(state, RX('theta', target_qubit), {'theta': jnp.pi})
```

We can also make any parametric gate controlled simply by passing a control qubit.

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux import RX, product_state, apply_gate

n_qubits = 2
target_qubit = 1
control_qubit = 0
state = product_state('11')
param_value = 1 / 4 * jnp.pi
new_state = apply_gate(state, RX(param_value, target_qubit, control_qubit))
```

## Analog Operations

`horqrux` also allows for global state evolution via the `HamiltonianEvolution` operation.
Note that it expects a hamiltonian and a time evolution parameter passed as `numpy` or `jax.numpy` arrays. To build arbitrary Pauli hamiltonians, we recommend using [Qadence](https://github.com/pasqal-io/qadence/blob/main/examples/backends/low_level/horqrux_analog.py).

```python exec="on" source="material-block"
from jax.numpy import pi, array, diag, kron, cdouble
from horqrux.analog import HamiltonianEvolution
from horqrux.apply import apply_gate
from horqrux.utils import uniform_state

sigmaz = diag(array([1.0, -1.0], dtype=cdouble))
Hbase = kron(sigmaz, sigmaz)

Hamiltonian = kron(Hbase, Hbase)
n_qubits = 4
t_evo = pi / 4
hamevo = HamiltonianEvolution(tuple([i for i in range(n_qubits)]))
psi = uniform_state(n_qubits)
psi_star = apply_gate(psi, hamevo, {"hamiltonian": Hamiltonian, "time_evolution": t_evo})
```

## Fitting a nonlinear function using adjoint differentiation

We can now build a fully differentiable variational circuit by simply defining a sequence of gates
and a set of initial parameter values we want to optimize.
`horqrux` provides an implementation of [adjoint differentiation](https://arxiv.org/abs/2009.02823),
which we can use to fit a function using a simple `Circuit` class.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

import jax
from jax import grad, jit, Array, value_and_grad, vmap
from dataclasses import dataclass
import jax.numpy as jnp
import optax
from functools import reduce, partial
from operator import add
from typing import Any, Callable
from uuid import uuid4

from horqrux.adjoint import adjoint_expectation
from horqrux.primitive import Primitive
from horqrux import Z, RX, RY, NOT, zero_state, apply_gate


n_qubits = 5
n_params = 3
n_layers = 3

# Lets define a sequence of rotations
def ansatz_w_params(n_qubits: int, n_layers: int) -> tuple[list, list]:
    all_ops = []
    param_names = []
    rots_fns = [RX ,RY, RX]
    for _ in range(n_layers):
        for i in range(n_qubits):
            ops = [fn(str(uuid4()), qubit) for fn, qubit in zip(rots_fns, [i for _ in range(len(rots_fns))])]
            param_names += [op.param for op in ops]
            ops += [NOT((i+1) % n_qubits, i % n_qubits) for i in range(n_qubits)]
            all_ops += ops

    return all_ops, param_names

#  We need a function to fit and use it to produce training data
fn = lambda x, degree: .05 * reduce(add, (jnp.cos(i*x) + jnp.sin(i*x) for i in range(degree)), 0)
x = jnp.linspace(0, 10, 100)
y = fn(x, 5)

@dataclass
class Circuit:
    n_qubits: int
    n_layers: int

    def __post_init__(self) -> None:
        # We will use a featuremap of RX rotations to encode some classical data
        self.feature_map: list[Primitive] = [RX('phi', i) for i in range(self.n_qubits)]
        self.ansatz, self.param_names = ansatz_w_params(self.n_qubits, self.n_layers)
        self.observable: list[Primitive] = [Z(0)]

    @partial(vmap, in_axes=(None, None, 0))
    def __call__(self, param_values: Array, x: Array) -> Array:
        state = zero_state(self.n_qubits)
        param_dict = {name: val for name, val in zip(self.param_names, param_values)}
        return adjoint_expectation(state, self.feature_map + self.ansatz, self.observable, {**param_dict, **{'phi': x}})


    @property
    def n_vparams(self) -> int:
        return len(self.param_names)

circ = Circuit(n_qubits, n_layers)
# Create random initial values for the parameters
key = jax.random.PRNGKey(42)
param_vals = jax.random.uniform(key, shape=(circ.n_vparams,))
# Check the initial predictions using randomly initialized parameters
y_init = circ(param_vals, x)

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(param_vals)

# Define a loss function
def loss_fn(param_vals: Array, x: Array, y: Array) -> Array:
    y_pred = circ(param_vals, x)
    return jnp.mean(optax.l2_loss(y_pred, y))


def optimize_step(param_vals: Array, opt_state: Array, grads: Array) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state)
    param_vals = optax.apply_updates(param_vals, updates)
    return param_vals, opt_state

@jit
def train_step(i: int, paramvals_w_optstate: tuple
) -> tuple:
    param_vals, opt_state = paramvals_w_optstate
    loss, grads = value_and_grad(loss_fn)(param_vals, x, y)
    param_vals, opt_state = optimize_step(param_vals, opt_state, grads)
    return param_vals, opt_state


n_epochs = 200
param_vals, opt_state = jax.lax.fori_loop(0, n_epochs, train_step, (param_vals, opt_state))
y_final = circ(param_vals, x)

# Lets plot the results
import matplotlib.pyplot as plt
plt.plot(x, y, label="truth")
plt.plot(x, y_init, label="initial")
plt.plot(x, y_final, "--", label="final", linewidth=3)
plt.legend()

from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
# from docs import docutils # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```
## Fitting a partial differential equation using DQC

Finally, we show how [DQC](https://arxiv.org/abs/2011.10395) can be implemented in `horqrux` and solve a partial differential equation.

```python exec="on" source="material-block" html="1"
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
from horqrux.primitive import Primitive
from horqrux.utils import inner

LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
X_POS = 0
Y_POS = 1
N_POINTS = 150
N_EPOCHS = 1000


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

    def __call__(self, state: Array, values: dict) -> Array:
        return reduce(add, [apply_gate(state, pauli, values) for pauli in self.paulis])


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

    def __call__(self, param_vals: Array, x: Array, y: Array) -> Array:
        state = zero_state(self.n_qubits)
        param_dict = {name: val for name, val in zip(self.param_names, param_vals)}
        out_state = apply_gate(
            state, self.feature_map + self.ansatz, {**param_dict, **{"x": x, "y": y}}
        )
        projected_state = self.observable(state, param_dict)
        return jnp.real(inner(out_state, projected_state))

    @property
    def n_vparams(self) -> int:
        return len(self.param_names)


circ = Circuit(N_QUBITS, DEPTH)
# Create random initial values for the parameters
key = jax.random.PRNGKey(42)
param_vals = jax.random.uniform(key, shape=(circ.n_vparams,))

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(param_vals)


def exp_fn(param_vals: Array, x: Array, y: Array) -> Array:
    return circ(param_vals, x, y)


def loss_fn(param_vals: Array, x: Array, y: Array) -> Array:
    def pde_loss(x: float, y: float) -> Array:
        l_b, r_b, t_b, b_b = list(
            map(
                lambda xy: exp_fn(param_vals, *xy),
                [
                    [jnp.zeros((1, 1)), y],  # u(0,y)=0
                    [jnp.ones((1, 1)), y],  # u(L,y)=0
                    [x, jnp.ones((1, 1))],  # u(x,H)=0
                    [x, jnp.zeros((1, 1))],  # u(x,0)=f(x)
                ],
            )
        )
        b_b -= jnp.sin(jnp.pi * x)
        hessian = jax.hessian(lambda xy: exp_fn(param_vals, xy[0], xy[1]))(
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
        interior = hessian[X_POS][X_POS] + hessian[Y_POS][Y_POS]  # uxx+uyy=0
        return reduce(add, list(map(lambda term: jnp.power(term, 2), [l_b, r_b, t_b, b_b, interior])))

    return jnp.mean(vmap(pde_loss, in_axes=(0, 0))(x, y))


def optimize_step(param_vals: Array, opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, param_vals)
    param_vals = optax.apply_updates(param_vals, updates)
    return param_vals, opt_state


# collocation points sampling and training
def sample_points(n_in: int, n_p: int) -> Array:
    return uniform(0, 1.0, (n_in, n_p))


@jit
def train_step(i: int, paramvals_w_optstate: tuple) -> tuple:
    param_vals, opt_state = paramvals_w_optstate
    x, y = sample_points(2, N_POINTS)
    loss, grads = value_and_grad(loss_fn)(param_vals, x, y)
    return optimize_step(param_vals, opt_state, grads)


param_vals, opt_state = jax.lax.fori_loop(0, N_EPOCHS, train_step, (param_vals, opt_state))
# compare the solution to known ground truth
single_domain = jnp.linspace(0, 1, num=N_POINTS)
domain = jnp.array(list(product(single_domain, single_domain)))
# analytical solution
analytic_sol = (
    (np.exp(-np.pi * domain[:, 0]) * np.sin(np.pi * domain[:, 1])).reshape(N_POINTS, N_POINTS).T
)
# DQC solution

dqc_sol = vmap(lambda domain: exp_fn(param_vals, domain[0], domain[1]), in_axes=(0,))(domain).reshape(
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
from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
# from docs import docutils # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```
