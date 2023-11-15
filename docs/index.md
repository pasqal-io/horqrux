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
from horqrux.gates import X, NOT
from horqrux.utils import prepare_state
from horqrux.ops import apply_gate

state = prepare_state(2)
new_state = apply_gate(state, X(0))
new_state= apply_gate(state, NOT(1,0))
```
