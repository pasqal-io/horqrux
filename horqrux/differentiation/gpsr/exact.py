from __future__ import annotations

from functools import partial, singledispatch
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array, random
from jax.experimental import checkify

from horqrux.apply import apply_operator
from horqrux.composite import Observable
from horqrux.primitives.primitive import Primitive
from horqrux.utils import DensityMatrix, State, expand_operator, num_qubits, inner
