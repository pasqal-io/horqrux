# Variational quantum eigensolver

We will demonstrate how to perform a Variational quantum eigensolver (VQE)[^1] task on a molecular example using Horqrux.
VQE boils down to finding the molecular ground state $| \psi(\theta) \rangle$ that minimizes the energy with respect to a molecular hamiltonian of interest denoted $H$ :
$$\langle \psi(\theta) | H | \psi(\theta) \rangle$$

## Hamiltonian

In this example, we run VQE for the $H2$ molecule in the STO-3G basis with a bondlength of $0.742 \mathring{A}$[^2]. The groud-state energy is around $-1.137$.
Note we need to manually create it by hand, as no syntax method is implemented (such methods are available in `Qadence` though).

```python exec="on" source="material-block" session="vqe"

import jax
from jax import Array
import optax

import time

import horqrux
from horqrux import I, Z, X, Y
from horqrux import Scale, Observable
from horqrux.composite import OpSequence

from horqrux.api import expectation
from horqrux.circuit import hea, QuantumCircuit

H2_hamiltonian = Observable([
  Scale(I(0), -0.09963387941370971),
  Scale(Z(0), 0.17110545123720233),
  Scale(Z(1), 0.17110545123720225),
  Scale(OpSequence([Z(0) , Z(1)]), 0.16859349595532533),
  Scale(OpSequence([Y(0) , X(1) , X(2) , Y(3)]), 0.04533062254573469),
  Scale( OpSequence([Y(0) , Y(1) , X(2) , X(3)]) , -0.04533062254573469),
  Scale( OpSequence([X(0) , X(1) , Y(2) , Y(3)]) , -0.04533062254573469),
  Scale( OpSequence([X(0) , Y(1) , Y(2) , X(3)]),  0.04533062254573469),
  Scale(Z(2),-0.22250914236600539),
  Scale( OpSequence([Z(0) , Z(2)]), 0.12051027989546245),
  Scale(Z(3), -0.22250914236600539),
  Scale(OpSequence([Z(0) , Z(3)]), 0.16584090244119712),
  Scale(OpSequence([Z(1) , Z(2)]), 0.16584090244119712),
  Scale(OpSequence([Z(1) , Z(3)]), 0.12051027989546245),
  Scale(OpSequence([Z(2) , Z(3)]), 0.1743207725924201),
])

```

## Ansatz

As an ansatz, we use the hardware-efficient ansatz[^3] with $5$ layers applied on the initial state $| 0011 \rangle$.

```python exec="on" source="material-block" session="vqe"

init_state = horqrux.product_state("0011")
ansatz = QuantumCircuit(4, hea(4, 5))
print("Number of variational parameters: ", ansatz.n_vparams) # markdown-exec: hide

```

## Optimization

The objective here is to optimize the variational parameters of our ansatz using the standard Adam optimizer. Below we show how to set up a train function.
We first consider the non-jitted version of the training function to compare later the timing with the jitted-version.

```python exec="on" source="material-block" session="vqe"
# Create random initial values for the parameters
key = jax.random.PRNGKey(42)
init_param_vals = jax.random.uniform(key, shape=(ansatz.n_vparams,))
LEARNING_RATE = 0.01
N_EPOCHS = 50

optimizer = optax.adam(learning_rate=LEARNING_RATE)

def optimize_step(param_vals: Array, opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, param_vals)
    param_vals = optax.apply_updates(param_vals, updates)
    return param_vals, opt_state

def loss_fn(param_vals: Array) -> Array:
    """The loss function is the sum of all expectation value for the observable components."""
    values = dict(zip(ansatz.vparams, param_vals))
    return jax.numpy.sum(expectation(init_state, ansatz, observables=[H2_hamiltonian], values=values))

print(f"Initial loss: {loss_fn(init_param_vals):.3f}") # markdown-exec: hide

def train_step(i: int, param_vals_opt_state: tuple) -> tuple:
    param_vals, opt_state = param_vals_opt_state
    loss, grads = jax.value_and_grad(loss_fn)(param_vals)
    return optimize_step(param_vals, opt_state, grads)

# set initial parameters and the state of the optimizer
param_vals = init_param_vals.clone()
opt_state = optimizer.init(init_param_vals)

def train_unjitted(param_vals, opt_state):
    for i in range(0, N_EPOCHS):
        param_vals, opt_state = train_step(i, (param_vals, opt_state))
    return param_vals, opt_state

start = time.time()
param_vals, opt_state = train_unjitted(param_vals, opt_state)
end = time.time()
time_nonjit = end - start

print(f"Final loss: {loss_fn(param_vals):.3f}") # markdown-exec: hide

```

Now, we will jit the `train_step` function with `jax.lax.fori_loop` and improve execution time (expecting at least $10$ times faster, depending on system):

```python exec="on" source="material-block" session="vqe"
# reset state and parameters
param_vals = init_param_vals.clone()
opt_state = optimizer.init(param_vals)

start_jit = time.time()
param_vals, opt_state = jax.lax.fori_loop(0, N_EPOCHS, train_step, (param_vals, opt_state))
end_jit = time.time()
time_jit = end_jit - start_jit

print(f"Time speedup: {time_nonjit / time_jit:.3f}")

```


[^1]: [Tilly et al., The Variational Quantum Eigensolver: a review of methods and best practices (2022)](https://arxiv.org/abs/2111.05176)
[^2]: [Pennylane, Quantum Datasets](https://docs.pennylane.ai/en/stable/introduction/data.html)
[^3]: [Kandala et al., Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets (2017)](https://doi.org/10.1038/nature23879)
