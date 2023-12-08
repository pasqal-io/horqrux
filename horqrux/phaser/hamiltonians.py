from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp
from chex import Array, PRNGKey

from .propagators import apply_unitary


class HamiltonianTerm(nn.Module):
    idx: tuple[int]
    weight_init_fn: Callable[[PRNGKey, tuple], Array] | None
    expm: Callable
    H: Callable

    def setup(self):
        if self.weight_init_fn is not None:
            self.weight = self.param("weight", self.weight_init_fn)
        else:
            self.weight = None

    def __call__(self, state, weight: Optional[Array] = None):
        # returns H |psi>
        if weight is None:
            weight = self.weight
        return apply_unitary(state, self.H(self.idx, weight), self.idx)

    def evolve(self, state, t: float, weight: Optional[Array] = None):
        # return expm(-iHt)|psi>
        if weight is None:
            weight = self.weight
        return apply_unitary(state, self.expm(self.idx, t * weight), self.idx)

    @classmethod
    def create(cls, H_fn, expm_fn):
        # Creates a specific hamiltonian.
        return partial(cls, expm=expm_fn, H=H_fn)


# Useful matrices
sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
sx = jnp.array([[0.0, 1.0], [1.0, 0.0]])
n = (sz + jnp.eye(2)) / 2


# Pauli z
def pauli_z_expm(idx, theta: float):
    # Implements expm(-1j * theta * sz)
    return jnp.cos(theta) * jnp.eye(2) - 1j * jnp.sin(theta) * sz


def pauli_z_H(idx, theta: float):
    return theta * sz


Pauli_z = HamiltonianTerm.create(pauli_z_H, pauli_z_expm)


# Pauli_x
def pauli_x_expm(idx, theta: float):
    # Implements expm(-1j * theta * sx)
    return jnp.cos(theta) * jnp.eye(2) - 1j * jnp.sin(theta) * sx


def pauli_x_H(idx, theta: float):
    return theta * sx


Pauli_x = HamiltonianTerm.create(pauli_x_H, pauli_x_expm)


# Number operator
def number_expm(idx, theta: float):
    return jnp.diag(jnp.exp(-1j * theta * jnp.array([1.0, 0.0])))


def number_H(idx, theta: float):
    return theta * n


Number = HamiltonianTerm.create(number_H, number_expm)


# Interaction operator
def interaction_expm(idx, u_ij: float):
    return jnp.diag(jnp.exp(-1j * u_ij * jnp.array([1.0, 0.0, 0.0, 0.0])))


def interaction_H(idx, u_ij: float):
    return u_ij * jnp.kron(n, n)


Interaction = HamiltonianTerm.create(interaction_H, interaction_expm)
