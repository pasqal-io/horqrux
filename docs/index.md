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
from horqrux.gates import X
from horqrux.utils import prepare_state
from horqrux.ops import apply_gate

state = prepare_state(2)
new_state = apply_gate(state, X(0))
```

We can also make any gate controlled, in the case of CNOT, we have to pass the target qubit first!

```python exec="on" source="material-block"
from horqrux.gates import NOT
from horqrux.utils import prepare_state
from horqrux.ops import apply_gate

state = prepare_state(2)
control = 0
target = 1
new_state= apply_gate(state, NOT(target,control))  # This performs
```

When applying parametric gates, we pass the numeric value for the parameter first

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux.gates import Rx
from horqrux.utils import prepare_state
from horqrux.ops import apply_gate

target_qubit = 1
state = prepare_state(target_qubit+1)
param_value = 1 / 4 * jnp.pi
new_state = apply_gate(state, Rx(param_value, target_qubit))
```