## Differentiation

`horqrux` also offers several differentiation modes to compute gradients which can be accessed through the
`expectation` API. It requires to pass one of the three `DiffMode` options to the `diff_mode` argument.
The default is `ad`.

### Automatic Differentiation (DiffMode.AD)
The default differentation mode of `horqrux` uses [jax.grad](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html), the `jax` native automatic differentiation engine which tracks operations on `jax.Array` objects by constructing a computational graph to perform chain rules for derivative calculations.

### Adjoint Differentiation (DiffMode.ADJOINT)
The [adjoint differentiation mode](https://arxiv.org/abs/2009.02823) computes first-order gradients by only requiring at most three states in memory in `O(P)` time where `P` is the number of parameters in a circuit.

### Generalized Parameter-Shift rules (DiffMode.GPSR)
The Generalized parameter shift rule (GPSR mode) is an extension of the well known [parameter shift rule (PSR)](https://arxiv.org/abs/1811.11184) algorithm [to arbitrary quantum operations](https://arxiv.org/abs/2108.01218). Indeed, PSR applies for quantum operations whose generator has a single gap in its eigenvalue spectrum. GPSR extends to multi-gap eigenvalued generators.

!!! warning "Usage restrictions"
    At the moment, circuits with one or more `Scale` and/or `HamiltonianEvolution` operations are not supported.
    They should be handled differently as GPSR requires operations to be of the form presented below.

For this, we define the differentiable function as quantum expectation value:

$$
f(x) = \left\langle 0\right|\hat{U}^{\dagger}(x)\hat{C}\hat{U}(x)\left|0\right\rangle
$$

where $\hat{U}(x)={\rm exp}{\left( -i\frac{x}{2}\hat{G}\right)}$ is the quantum evolution operator with generator $\hat{G}$ representing the structure of the underlying quantum circuit and $\hat{C}$ is the cost operator. Then using the eigenvalue spectrum $\left\{ \lambda_n\right\}$ of the generator $\hat{G}$ we calculate the full set of corresponding unique non-zero spectral gaps $\left\{ \Delta_s\right\}$ (differences between eigenvalues). It can be shown that the final expression of derivative of $f(x)$ is then given by the following expression:

$\begin{equation}
\frac{{\rm d}f\left(x\right)}{{\rm d}x}=\overset{S}{\underset{s=1}{\sum}}\Delta_{s}R_{s},
\end{equation}$

where $S$ is the number of unique non-zero spectral gaps and $R_s$ are real quantities that are solutions of a system of linear equations

$\begin{equation}
\begin{cases}
F_{1} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{1}\Delta_{s}}{2}\right)R_{s},\\
F_{2} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{2}\Delta_{s}}{2}\right)R_{s},\\
 & ...\\
F_{S} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{M}\Delta_{s}}{2}\right)R_{s}.
\end{cases}
\end{equation}$

Here $F_s=f(x+\delta_s)-f(x-\delta_s)$ denotes the difference between values of functions evaluated at shifted arguments $x\pm\delta_s$.


## Examples

### Circuit parameters differentiation

We show below a code example with several differentiation methods for circuit parameters.
Note that [jax.grad](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html) requires functions of `Array`.

```python exec="on" source="material-block" html="1" session="diff"

import jax
import jax.numpy as jnp
from jax import Array

from horqrux import expectation, random_state, DiffMode
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.parametric import RX
from horqrux.primitives.primitive import Z

N_QUBITS = 2

x = jax.random.uniform(jax.random.key(0), (2,))

param_prefix = "theta"
param_names = [param_prefix, param_prefix + "2"]
ops = [RX(param_names[0], 0), RX(param_names[1], 1)]

def values_to_dict(x: Array) -> dict[str, Array]:
    return {param_names[0]: x[0], param_names[1]: x[1]}

circuit = QuantumCircuit(2, ops)
observables = [Observable([Z(0)]), Observable([Z(1)])]
state = random_state(N_QUBITS)

def expectation_ad(x: Array) -> Array:
    values = values_to_dict(x)
    return expectation(state, circuit, observables, values, diff_mode=DiffMode.AD).sum()

def expectation_gpsr(x: Array) -> Array:
    values = values_to_dict(x)
    return expectation(state, circuit, observables, values, diff_mode=DiffMode.GPSR).sum()

def expectation_adjoint(x: Array) -> Array:
    values = values_to_dict(x)
    return expectation(state, circuit, observables, values, diff_mode= DiffMode.ADJOINT).sum()

d_ad = jax.grad(expectation_ad)
d_gpsr = jax.grad(expectation_gpsr)
d_adjoint = jax.grad(expectation_adjoint)

grad_ad = d_ad(x)
grad_gpsr = d_gpsr(x)
grad_adjoint = d_adjoint(x)
print(f"Gradient: {grad_ad}") # markdown-exec: hide
```

### Parametrized observable differentiation

To allow differentiating observable parameters only, we need to specify the `values` argument as a dictionary with two keys `circuit` and `observables`, each being a dictionary of corresponding parameters and values, as follows:

```python exec="on" source="material-block" html="1" session="diff"

from horqrux.primitives.parametric import RZ
observables = [Observable([RZ(param_prefix + "_obs", 0)])]
obsval = jax.random.uniform(jax.random.key(0), (1,))


def expectation_separate_parameters(x: Array, y: Array) -> Array:
    values = {"circuit": values_to_dict(x), "observables": {param_prefix + "_obs": y}}
    return expectation(state, circuit, observables, values, diff_mode=DiffMode.AD).sum()

dobs_ad = jax.grad(expectation_separate_parameters, argnums=1)
grad_ad = dobs_ad(x, obsval)
print(f"Gradient: {grad_ad}") # markdown-exec: hide
```
