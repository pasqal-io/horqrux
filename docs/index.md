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
from horqrux.primitive import X
from horqrux.utils import random_state
from horqrux.apply import apply_gate

state = random_state(2)
new_state = apply_gate(state, X(0))
```

We can also make any gate controlled, in the case of X, we have to pass the target qubit first!

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux.primitive import X
from horqrux.utils import product_state, equivalent_state
from horqrux.apply import apply_gate

n_qubits = 2
state = product_state('11')
control = 0
target = 1
# This is equivalent to performing CNOT(0,1)
new_state= apply_gate(state, X(target,control))
assert jnp.allclose(new_state, product_state('10'))
```

When applying parametric gates, we pass the numeric value for the parameter first

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux.parametric import RX
from horqrux.utils import random_state
from horqrux.apply import apply_gate

target_qubit = 1
state = random_state(target_qubit+1)
param_value = 1 / 4 * jnp.pi
new_state = apply_gate(state, RX(param_value, target_qubit))
```

We can also make any parametric gate controlled simply by passing a control qubit.

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux.parametric import RX
from horqrux.utils import product_state
from horqrux.apply import apply_gate

n_qubits = 2
target_qubit = 1
control_qubit = 0
state = product_state('11')
param_value = 1 / 4 * jnp.pi
new_state = apply_gate(state, RX(param_value, target_qubit, control_qubit))
```

A fully differentiable variational circuit is simply a sequence of gates which are applied to a state.

```python exec="on" source="material-block" html="1"
import jax
from jax import grad, jit, Array, value_and_grad, vmap
import jax.numpy as jnp
import optax
from itertools import chain
from functools import reduce, partial
from operator import add
from typing import Callable
import matplotlib.pyplot as plt
from horqrux.abstract import Operator
from horqrux import Z, RX, RY, NOT
from horqrux.utils import zero_state, overlap
from horqrux.apply import apply_gate

n_qubits = 5
# Lets define a sequence of rotations
n_params = 3
n_layers = 3
param_names = [f'theta_{i}' for i in range(n_params * n_layers * n_qubits)]
qubits = [q for _ in range(n_params) for q in range(n_qubits)  for _ in range(n_layers)]

rots = [(RX,RY,RX) for _ in range(n_layers * n_qubits)]
# breakpoint()
ops = [fn(param, qubit) for fn, param, qubit in zip([item for tple in rots for item in tple], param_names, qubits)]
# Create random initial values for the parameters
key = jax.random.PRNGKey(42)
param_vals = jax.random.uniform(key, shape=(len(ops),))
param_dict = {name: val for name, val in zip(param_names, param_vals)}
# We will use a featuremap of RX rotations to encode some classical data
x = jnp.linspace(0, 10, 100)
# We need a function which runs our circuit
def circ(param_dict: dict[str, float], rotations: list[Operator] = ops, n_qubits: int=n_qubits) -> jax.Array:
    feature_map = [RX('phi', i) for i in range(n_qubits)]
    entangling = [NOT((i+1) % n_qubits, i % n_qubits) for i in range(n_qubits)]
    observable = [Z(i) for i in range(n_qubits)]
    state = apply_gate(zero_state(n_qubits), feature_map + rotations + entangling, param_dict)
    projection = apply_gate(state, observable)
    return overlap(state, projection)

# Lets create a convenience lambda fn to use for forward passes.
expfn = lambda p, v: circ({**p, **{'phi': v}})

# Check the initial predictions using randomly initialized parameters
y_init = vmap(partial(expfn, param_dict), in_axes=(0,))(x)
# Let's compute both values and gradients for a set of parameters.
expval_and_grads = value_and_grad(lambda p: expfn(p, 1.))(param_dict)

# We can also train our model to fit a function

fn = lambda x, degree: .05 * reduce(add, (jnp.cos(i*x) + jnp.sin(i*x) for i in range(degree)), 0)
DEGREE = 5
y = fn(x, DEGREE)

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(param_dict)


def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def loss_fn(params: dict, v: float) -> float:
    return (expfn(params, v) - fn(v, DEGREE)) ** 2

@jit
def train_step(i:int, inputs: tuple
) -> tuple:
    params, opt_state = inputs
    def vng(v: float) -> tuple:
        return value_and_grad(partial(loss_fn, v=v))(param_dict)
    loss, grads = vmap(vng, in_axes=(0,))(x)
    loss, grads = jnp.mean(loss), {param: jnp.mean(grad_vals) for param, grad_vals in grads.items()}
    params, opt_state = optimize_step(params, opt_state, grads)
    print(f"epoch {i} loss:{loss}")
    return params, opt_state

n_epochs = 1000
param_dict, opt_state = jax.lax.fori_loop(0, n_epochs, train_step, (param_dict, opt_state))
y_final = vmap(partial(expfn, param_dict), in_axes=(0,))(x)

# Lets plot the results
plt.plot(x, y, label="truth")
plt.plot(x, y_init, label="initial")
plt.plot(x, y_final, "--", label="final", linewidth=3)
plt.legend()
# from docs.docsutils import fig_to_html # markdown-exec: hide
# print(fig_to_html(plt.gcf())) # markdown-exec: hide
```
