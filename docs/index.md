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

We can also make any gate controlled, in the case of X, we have to pass the target qubit first!

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux.gates import X
from horqrux.utils import prepare_state, equivalent_state
from horqrux.ops import apply_gate

n_qubits = 2
state = prepare_state(n_qubits, '11')
control = 0
target = 1
# This is equivalent to performing CNOT(0,1)
new_state= apply_gate(state, X(target,control))
assert jnp.allclose(new_state, prepare_state(n_qubits, '10'))
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

We can also make any parametric gate controlled simply by passing a control qubit.

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux.gates import Rx
from horqrux.utils import prepare_state
from horqrux.ops import apply_gate

n_qubits = 2
target_qubit = 1
control_qubit = 0
state = prepare_state(2, '11')
param_value = 1 / 4 * jnp.pi
new_state = apply_gate(state, Rx(param_value, target_qubit, control_qubit))
```

A fully differentiable variational circuit is simply a sequence of gates which are applied to a state.

```python exec="on" source="material-block"
import jax
import jax.numpy as jnp
from horqrux import gates
from horqrux.utils import prepare_state, overlap
from horqrux.ops import apply_gate

n_qubits = 2
state = prepare_state(2, '00')
# Lets define a sequence of rotations
ops = [gates.Rx, gates.Ry, gates.Rx]
# Create random initial values for the parameters
key = jax.random.PRNGKey(0)
params = jax.random.uniform(key, shape=(n_qubits * len(ops),))

def circ(state) -> jax.Array:
    for qubit in range(n_qubits):
        for gate,param in zip(ops, params):
            state = apply_gate(state, gate(param, qubit))
    state = apply_gate(state,gates.NOT(1, 0))
    projection = apply_gate(state, gates.Z(0))
    return overlap(state, projection)

# Let's compute both values and gradients for a set of parameters and compile the circuit.
circ = jax.jit(jax.value_and_grad(circ))
# Run it on a state.
expval_and_grads = circ(state)
expval = expval_and_grads[0]
grads = expval_and_grads[1:]
print(f'Expval: {expval};'
       f'Grads: {grads}')
```
