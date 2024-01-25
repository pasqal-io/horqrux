# Welcome to horqrux

**horqrux** is a state vector simulator designed for quantum machine learning written in [JAX](https://jax.readthedocs.io/).

## Setup

To install `horqrux` , you can go into any virtual environment of your
choice and install it normally with `pip`:

```bash
pip install horqrux
```

## Gates

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

We can now build a fully differentiable variational circuit by simply defining a sequence of gates
and a set of initial parameter values we want to optimize.
Lets fit a function using a simple circuit class wrapper.

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

from horqrux.abstract import Operator
from horqrux import Z, RX, RY, NOT, zero_state, apply_gate, overlap


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
        self.feature_map: list[Operator] = [RX('phi', i) for i in range(n_qubits)]
        self.ansatz, self.param_names = ansatz_w_params(self.n_qubits, self.n_layers)
        self.observable: list[Operator] = [Z(0)]

    @partial(vmap, in_axes=(None, None, 0))
    def forward(self, param_values: Array, x: Array) -> Array:
        state = zero_state(self.n_qubits)
        param_dict = {name: val for name, val in zip(self.param_names, param_values)}
        state = apply_gate(state, self.feature_map + self.ansatz, {**param_dict, **{'phi': x}})
        return overlap(state, apply_gate(state, self.observable))

    def __call__(self, param_values: Array, x: Array) -> Array:
        return self.forward(param_values, x)

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


def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

@jit
def train_step(i: int, inputs: tuple
) -> tuple:
    param_vals, opt_state = inputs
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
