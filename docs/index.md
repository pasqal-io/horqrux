# Welcome to horqrux

**horqrux** is a state vector and density matrix simulator designed for quantum machine learning written in [JAX](https://jax.readthedocs.io/).

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
from horqrux import X, random_state, apply_gates

state = random_state(2)
new_state = apply_gates(state, X(0))
```

We can also make any gate controlled, in the case of X, we have to pass the target qubit first!

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux import X, product_state, apply_gates

n_qubits = 2
state = product_state('11')
control = 0
target = 1
# This is equivalent to performing CNOT(0,1)
new_state= apply_gates(state, X(target,control))
assert jnp.allclose(new_state, product_state('10'))
```

When applying parametric gates, we can either pass a numeric value or a parameter name for the parameter as the first argument.

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux import RX, random_state, apply_gates

target_qubit = 1
state = random_state(target_qubit+1)
param_value = 1 / 4 * jnp.pi
new_state = apply_gates(state, RX(param_value, target_qubit))
# Parametric horqrux gates also accept parameter names in the form of strings.
# Simply pass a dictionary of parameter names and values to the 'apply_gates' function
new_state = apply_gates(state, RX('theta', target_qubit), {'theta': jnp.pi})
```

We can also make any parametric gate controlled simply by passing a control qubit.

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux import RX, product_state, apply_gates

n_qubits = 2
target_qubit = 1
control_qubit = 0
state = product_state('11')
param_value = 1 / 4 * jnp.pi
new_state = apply_gates(state, RX(param_value, target_qubit, control_qubit))
```

### Using sparse matrices

`horqrux` also provide the possibility to use sparse matrices when performing operations using [Batched-coordinate (BCOO) sparse matrices](https://docs.jax.dev/en/latest/jax.experimental.sparse.html#batched-coordinate-bcoo-sparse-matrices). Note though that Jax's sparse matrices are still considered an experimental feature. For this, the input state and all operations should be initialized with `sparse=True`. One can also perform the sparse conversion of operators using the `to_sparse` method. And one can perform the reverse conversion using the `to_dense` method.

```python exec="on" source="material-block"
import jax.numpy as jnp
from horqrux import RX, product_state, apply_gates
from horqrux.utils.conversion import to_sparse, to_dense

n_qubits = 2
target_qubit = 1
control_qubit = 0
state = product_state('11', sparse=True)
param_value = 1 / 4 * jnp.pi
gate = to_sparse(RX(param_value, target_qubit, control_qubit))
new_state = apply_gates(state, gate)

gate = to_dense(RX(param_value, target_qubit, control_qubit))
state = to_dense(state)
new_state = apply_gates(state, gate)
```

!!! warning "Experimental Sparse matrices scope"
    Note this is an experimental feature (raise an issue if needed).
    We only support noiseless state-vector simulation with digital operations when using sparse matrices.
    Note that `jax.grad` or `jax.experimental.sparse.grad` does not work with sparse expectation operations.

## Analog Operations

`horqrux` also allows for global state evolution via the `HamiltonianEvolution` operation.
Note that it expects a hamiltonian and a time evolution parameter passed as `numpy` or `jax.numpy` arrays. To build arbitrary Pauli hamiltonians, we recommend using [Qadence](https://github.com/pasqal-io/qadence/blob/main/examples/backends/low_level/horqrux_analog.py).

```python exec="on" source="material-block"
from jax.numpy import pi, array, diag, kron, cdouble
from horqrux.analog import HamiltonianEvolution
from horqrux.apply import apply_gates
from horqrux.utils.operator_utils import uniform_state

sigmaz = diag(array([1.0, -1.0], dtype=cdouble))
Hbase = kron(sigmaz, sigmaz)

Hamiltonian = kron(Hbase, Hbase)
n_qubits = 4
t_evo = pi / 4
hamevo = HamiltonianEvolution(tuple([i for i in range(n_qubits)]))
psi = uniform_state(n_qubits)
psi_star = apply_gates(psi, hamevo, {"hamiltonian": Hamiltonian, "time_evolution": t_evo})
```

## Circuits

Using digital and analog operations, you can can build fully differentiable quantum circuits using the `QuantumCircuit` class.

```python exec="on" source="material-block" html="1"
import jax.numpy as jnp
from horqrux import Z, RX, RY, NOT, expectation
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.utils.operator_utils import zero_state

ops = [RX("theta", 0), RY("epsilon", 0), RX("phi", 0), NOT(1, 0), RX("omega", 0, 1)]
circuit = QuantumCircuit(2, ops)
observable = [Observable([Z(0)])]
values = {
    "theta": 0.2,
    "epsilon": 0.3,
    "phi": 0.4,
    "omega": 0.5,
}
state = zero_state(2)
exp_qc = expectation(state, circuit, observable, values)
```
