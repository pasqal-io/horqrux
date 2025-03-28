# Fitting a nonlinear function using adjoint differentiation

We can build a fully differentiable variational circuit by defining a sequence of gates
and a set of optimizable parameter values.
`horqrux` provides an implementation for the adjoint differentiation method[^1],
which we can use to fit a function using a simple `FunctionFitter` class inheriting from the `QuantumCircuit` class.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

import jax
from jax import grad, jit, Array, value_and_grad, vmap
from jax.tree_util import register_pytree_node_class

from dataclasses import dataclass
import jax.numpy as jnp
import optax
from functools import reduce, partial
from operator import add
from typing import Any, Callable
from uuid import uuid4

from horqrux import expectation, Observable
from horqrux import Z, RX, RY, NOT, zero_state, apply_gates
from horqrux.circuit import QuantumCircuit, hea
from horqrux.primitives.primitive import Primitive
from horqrux.primitives.parametric import Parametric
from horqrux.utils.operator_utils import DiffMode


n_qubits = 5
n_params = 3
n_layers = 3

#  We need a target function to fit and to produce training data
fn = lambda x, degree: .05 * reduce(add, (jnp.cos(i*x) + jnp.sin(i*x) for i in range(degree)), 0)
x = jnp.linspace(0, 10, 100)
y = fn(x, 5)

class FunctionFitter(QuantumCircuit):
    """
    The FunctionFitter is composed of a quantum circuit and an observable to obtain a real-valued output.
    It can be seen as a function of input values `x` that are passed as parameter values of a subset of parameterized quantum gates.
    The output is define as
    The rest of the parameterized quantum gates use the `values` coming from a classical optimizer.

    Attributes:
        n_qubits (int): Number of qubits.
        operations (list[Primitive]): Operations defining the circuit.
        fparams (list[str]): List of parameters that are considered
            non trainable, used for passing fixed input data to a quantum circuit.
            The corresponding operations compose the `feature map`.
        observable (Observable): Observable for getting real-valued measurement output.
            Here, we use the Z observable applied on qubit 0.
        state (Array): Initial zero state.
    """
    def __init__(self, n_qubits, operations, fparams) -> None:
        super().__init__(n_qubits, operations, fparams)
        self.observable: Observable = Observable([Z(0)])
        self.state = zero_state(self.n_qubits)

    @partial(vmap, in_axes=(None, None, 0))
    def __call__(self, param_values: Array, x: Array) -> Array:
        param_dict = {name: val for name, val in zip(self.vparams, param_values)}
        return jnp.squeeze(expectation(self.state, self, [self.observable], {**param_dict, **{'phi': x}}, DiffMode.ADJOINT))

feature_map = [RX('phi', i) for i in range(n_qubits)]
fm_names = [f.param for f in feature_map]
ansatz = hea(n_qubits, n_layers)
circ = FunctionFitter(n_qubits, feature_map + ansatz, fm_names)
# Create random initial values for the parameters
key = jax.random.PRNGKey(42)
param_vals = jax.random.uniform(key, shape=(circ.n_vparams,))
# Check the initial predictions using randomly initialized parameters
y_init = circ(param_vals, x)

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(param_vals)

# Define a loss function
def loss_fn(param_vals: Array) -> Array:
    y_pred = circ(param_vals, x)
    return jnp.mean(optax.l2_loss(y_pred, y))


def optimize_step(param_vals: Array, opt_state: Array, grads: Array) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state)
    param_vals = optax.apply_updates(param_vals, updates)
    return param_vals, opt_state

@jit
def train_step(i: int, paramvals_w_optstate: tuple) -> tuple:
    param_vals, opt_state = paramvals_w_optstate
    loss, grads = value_and_grad(loss_fn)(param_vals)
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

# Fitting a partial differential equation using DifferentiableQuantumCircuit

We show how a Differentiable Quantum Circuit (DQC)[^2] can be implemented in `horqrux` and solve a partial differential equation.
To do so, we define a `PDESolver` class inheriting from the `QuantumCircuit` class.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import add
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array, jit, value_and_grad, vmap
from numpy.random import uniform

from horqrux.apply import group_by_index
from horqrux.circuit import QuantumCircuit, hea
from horqrux import NOT, RX, RY, Z, apply_gates, zero_state, Observable
from horqrux.primitives.primitive import Primitive
from horqrux.primitives.parametric import Parametric
from horqrux.utils.operator_utils import inner

LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
NUM_VARIABLES = len(VARIABLES)
X_POS, Y_POS = [i for i in range(NUM_VARIABLES)]
BATCH_SIZE = 500
N_EPOCHS = 500

def total_magnetization(n_qubits:int) -> Callable:
    paulis = Observable([Z(i) for i in range(n_qubits)])

    def _total_magnetization(out_state: Array, values: dict[str, Array]) -> Array:
        projected_state = paulis.forward(out_state, values)
        return inner(out_state, projected_state).real
    return _total_magnetization

class PDESolver(QuantumCircuit):
    """
    The PDESolver is composed of a quantum circuit and an observable to obtain a real-valued output.
    It can be seen as a function of input values `x`, `y` that are passed as parameter values of a subset of parameterized quantum gates.
    The rest of the parameterized quantum gates use the `values` coming from a classical optimizer.

    Attributes:
        n_qubits (int): Number of qubits.
        operations (list[Primitive]): Operations defining the circuit.
        fparams (list[str]): List of parameters that are considered
            non trainable, used for passing fixed input data to a quantum circuit.
                The corresponding operations compose the `feature map`.
        observable (Callable): Function applying the observable for getting real-valued output.
            Here, we use the total_magnetization function applying the observable \sum_i^N Z(i).
        state (Array): Initial zero state.
    """
    def __init__(self, n_qubits, operations, fparams) -> None:
        operations = group_by_index(operations)
        super().__init__(n_qubits, operations, fparams)
        self.observable = total_magnetization(self.n_qubits)
        self.state = zero_state(self.n_qubits)

    def __call__(self, values: dict[str, Array], x: Array, y: Array) -> Array:
        param_dict = {name: val for name, val in zip(self.vparams, values)}
        out_state = apply_gates(
            self.state, self.operations, {**param_dict, **{"f_x": x, "f_y": y}}
        )
        return self.observable(out_state, {})


fm =  [RX("f_x", i) for i in range(N_QUBITS // 2)] + [
            RX("f_y", i) for i in range(N_QUBITS // 2, N_QUBITS)
        ]
fm_circuit_parameters = [f.param for f in fm]
ansatz = hea(N_QUBITS, DEPTH)
circ = PDESolver(N_QUBITS, fm + ansatz, fm_circuit_parameters)
# Create random initial values for the parameters
key = jax.random.PRNGKey(42)
param_vals = jax.random.uniform(key, shape=(circ.n_vparams,))

optimizer = optax.adam(learning_rate=LEARNING_RATE)
opt_state = optimizer.init(param_vals)


def loss_fn(param_vals: Array) -> Array:
    def pde_loss(x: Array, y: Array) -> Array:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        left = (jnp.zeros_like(y), y)  # u(0,y)=0
        right = (jnp.ones_like(y), y)  # u(L,y)=0
        top = (x, jnp.ones_like(x))  # u(x,H)=0
        bottom = (x, jnp.zeros_like(x))  # u(x,0)=f(x)
        terms = jnp.dstack(list(map(jnp.hstack, [left, right, top, bottom])))
        loss_left, loss_right, loss_top, loss_bottom = vmap(lambda xy: circ(param_vals, xy[:, 0], xy[:, 1]), in_axes=(2,))(
            terms
        )
        loss_bottom -= jnp.sin(jnp.pi * x)
        hessian = jax.hessian(lambda xy: circ(param_vals, xy[0], xy[1]))(
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
        loss_interior = hessian[X_POS][X_POS] + hessian[Y_POS][Y_POS]  # uxx+uyy=0
        return jnp.sum(
            jnp.concatenate(
                list(
                    map(
                        lambda term: jnp.power(term, 2).reshape(-1, 1),
                        [loss_left, loss_right, loss_top, loss_bottom, loss_interior],
                    )
                )
            )
        )

    return jnp.mean(vmap(pde_loss, in_axes=(0, 0))(*uniform(0, 1.0, (NUM_VARIABLES, BATCH_SIZE))))


def optimize_step(param_vals: Array, opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, param_vals)
    param_vals = optax.apply_updates(param_vals, updates)
    return param_vals, opt_state


@jit
def train_step(i: int, paramvals_w_optstate: tuple) -> tuple:
    param_vals, opt_state = paramvals_w_optstate
    loss, grads = value_and_grad(loss_fn)(param_vals)
    return optimize_step(param_vals, opt_state, grads)


param_vals, opt_state = jax.lax.fori_loop(0, N_EPOCHS, train_step, (param_vals, opt_state))
# compare the solution to known ground truth
single_domain = jnp.linspace(0, 1, num=BATCH_SIZE)
domain = jnp.array(list(product(single_domain, single_domain)))
# analytical solution
analytic_sol = (
    (np.exp(-np.pi * domain[:, 0]) * np.sin(np.pi * domain[:, 1])).reshape(BATCH_SIZE, BATCH_SIZE).T
)
# DQC solution
dqc_sol = vmap(lambda domain: circ(param_vals, domain[0], domain[1]), in_axes=(0,))(
    domain
).reshape(BATCH_SIZE, BATCH_SIZE)
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

[^1]: [Tyson Jones, Julien Gacon
, Efficient calculation of gradients in classical simulations of variational quantum algorithms
 (2020)](https://arxiv.org/abs/2011.10395)

[^2]: [Oleksandr Kyriienko, Annie E. Paine, Vincent E. Elfving, Solving nonlinear differential equations with differentiable quantum circuits
 (2020)](https://arxiv.org/abs/2011.10395)
