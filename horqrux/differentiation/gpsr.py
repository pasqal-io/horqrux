from __future__ import annotations

from functools import partial, singledispatch
from operator import is_not
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array, random

from horqrux.apply import apply_gates
from horqrux.composite import Observable
from horqrux.differentiation.ad import _ad_expectation_single_observable
from horqrux.primitives import Parametric, Primitive
from horqrux.utils import DensityMatrix, State, expand_operator, num_qubits


def is_parametric(gate: Primitive) -> bool:
    return isinstance(gate, Parametric) and isinstance(gate.param, str)


@singledispatch
def eigen_probabilities(state: Any, eigvecs: Array) -> Array:
    """Obtain the probabilities using an input state and the eigenvectors decomposition
       of an observable.

    Args:
        state (Any): Input.
        eigvecs (Array): Eigenvectors of the observables.

    Returns:
        Array: The probabilities.
    """
    raise NotImplementedError(
        f"eigen_probabilities is not implemented for the state type {type(state)}."
    )


@eigen_probabilities.register
def _(state: Array, eigvecs: Array) -> Array:
    """Obtain the probabilities using an input quantum state vector
        and the eigenvectors decomposition
        of an observable.

    Args:
        state (Array): Input array.
        eigvecs (Array): Eigenvectors of the observables.

    Returns:
        Array: The probabilities.
    """
    inner_prod = jnp.matmul(jnp.conjugate(eigvecs.T), state.flatten())
    return jnp.abs(inner_prod) ** 2


@eigen_probabilities.register
def _(state: DensityMatrix, eigvecs: Array) -> Array:
    """Obtain the probabilities using an input quantum density matrix
        and the eigenvectors decomposition
        of an observable.

    Args:
        state (DensityMatrix): Input density matrix.
        eigvecs (Array): Eigenvectors of the observables.

    Returns:
        Array: The probabilities.
    """
    mat_prob = jnp.conjugate(eigvecs.T) @ state.array @ eigvecs
    return mat_prob.diagonal().real


def eigen_sample(
    state: State,
    observables: list[Observable],
    values: dict[str, float],
    n_qubits: int,
    n_shots: int,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """Sample eigenvalues of observable given the probability distribution
        defined by applying the eigenvectors to the state.

    Args:
        state (State): Input state or density matrix.
        observables (list[Observable]): list of observables.
        values (dict[str, float]): Parameter values.
        n_qubits (int): Number of qubits
        n_shots (int): Number of samples
        key (Any, optional): Random seed key. Defaults to jax.random.PRNGKey(0).

    Returns:
        Array: Sampled eigenvalues.
    """
    qubits = tuple(range(n_qubits))
    d = 2**n_qubits
    mat_obs = list(
        map(
            lambda observable: expand_operator(
                observable.tensor(values), observable.qubit_support, qubits
            ).reshape((d, d)),
            observables,
        )
    )
    eigs = jax.vmap(jnp.linalg.eigh)(jnp.stack(mat_obs))
    eigvecs, eigvals = align_eigenvectors(eigs.eigenvalues, eigs.eigenvectors)
    probs = eigen_probabilities(state, eigvecs)
    return jax.random.choice(key=key, a=eigvals, p=probs, shape=(n_shots,)).mean(axis=0)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def no_shots_fwd(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable.
    """
    outputs = list(
        map(
            lambda observable: _ad_expectation_single_observable(
                apply_gates(state, gates, values),
                observable,
                values,
            ),
            observables,
        )
    )
    return jnp.stack(outputs)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 4, 5))
def finite_shots_fwd(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """
    Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectation given an observable and `n_shots` shots.
    """
    output_gates = apply_gates(state, gates, values)
    n_qubits = num_qubits(output_gates)
    if isinstance(state, DensityMatrix):
        d = 2**n_qubits
        output_gates.array = output_gates.array.reshape((d, d))
    return eigen_sample(output_gates, observables, values, n_qubits, n_shots, key)


def align_eigenvectors(eigenvalues: Array, eigenvectors: Array) -> tuple[Array, Array]:
    """
    Given a list of eigenvalue eigenvector matrix tuples in the form of
    [(eigenvalue, eigenvector)...], this function aligns all the eigenvector
    matrices so that they are identical, and also rearranges the corresponding
    eigenvalues.

    This is primarily used as a utility function to help sample multiple
    correlated observables when using finite shots.

    Given two permuted eigenvector matrices, A and B, we wish to find a permutation
    matrix P such that A P = B. This function calculates such a permutation
    matrix and uses it to align each eigenvector matrix to the first eigenvector
    matrix of eigs.
    """
    eigenvector_matrix = eigenvectors[0]

    P = jax.vmap(lambda mat: permutation_matrix(mat, eigenvector_matrix))(eigenvectors)
    # checkify.check(
    #     jnp.all(jax.vmap(validate_permutation_matrix)(P)),
    #     "Did not calculate valid permutation matrix",
    # )
    eigenvalues_aligned = jax.vmap(jnp.dot)(eigenvalues, P).T
    return eigenvector_matrix, eigenvalues_aligned


def permutation_matrix(mat: Array, eigenvector_matrix: Array) -> Array:
    return (jnp.linalg.inv(mat) @ eigenvector_matrix).real > 0.5


def validate_permutation_matrix(P: Array) -> Array:
    rows = P.sum(axis=0)
    columns = P.sum(axis=1)
    ones = jnp.ones(P.shape[0], dtype=rows.dtype)
    return ((ones == rows) & (ones == columns)).min()


@finite_shots_fwd.defjvp
def finite_shots_jvp(
    state: Array,
    gates: Union[Primitive, Iterable[Primitive]],
    observable: list[Observable],
    n_shots: int,
    key: Array,
    primals: tuple[dict[str, float]],
    tangents: tuple[dict[str, float]],
) -> Array:
    values = primals[0]
    tangent_dict = tangents[0]

    # TODO: compute spectral gap through the generator which is associated with
    # a param name.
    spectral_gap = 2.0
    shift = jnp.pi / 2

    def jvp_component(param_name: str, key: Array) -> Array:
        up_key, down_key = random.split(key)
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = finite_shots_fwd(state, gates, observable, up_val, n_shots, up_key)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = finite_shots_fwd(state, gates, observable, down_val, n_shots, down_key)
        grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        return grad

    params_with_keys = zip(values.keys(), random.split(key, len(values)))
    val_keys = tuple(values.keys())
    fwd = finite_shots_fwd(state, gates, observable, values, n_shots, key)
    jvp_caller = jvp_component
    if not isinstance(gates, Primitive):
        param_to_gates: dict[str, tuple] = dict.fromkeys(val_keys, tuple())
        for gate in gates:
            if is_parametric(gate) and gate.param in val_keys:  # type: ignore[attr-defined]
                param_to_gates[gate.param] += (gate,)  # type: ignore[attr-defined]

        if max(map(len, param_to_gates.values())) > 1:

            def jvp_component_repeated_param(param_name: str, key: Array) -> Array:
                shift_gates = param_to_gates[param_name]
                keys = random.split(key, len(shift_gates))

                def shift_jvp(gate: Parametric, key: Array) -> Array:
                    up_key, down_key = random.split(key)
                    f_up = finite_shots_fwd(state, gates, observable, values, n_shots, up_key)
                    f_down = finite_shots_fwd(state, gates, observable, values, n_shots, down_key)
                    grad = (
                        spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                    )
                    return grad

                return sum(
                    shift_jvp(shift_gate, key_gate)
                    for shift_gate, key_gate in zip(shift_gates, keys)
                )

            jvp_caller = jvp_component_repeated_param

    jvp = sum(jvp_caller(param, key) * tangent_dict[param] for param, key in params_with_keys)
    return fwd, jvp.reshape(fwd.shape)


# @no_shots_fwd.defjvp
# def no_shots_fwd_jvp(
#     state: Array,
#     gates: Union[Primitive, Iterable[Primitive]],
#     observable: list[Observable],
#     primals: tuple[dict[str, float]],
#     tangents: tuple[dict[str, float]],
# ) -> Array:
#     values = primals[0]
#     tangent_dict = tangents[0]

#     # TODO: compute spectral gap through the generator which is associated with
#     # a param name.
#     spectral_gap = 2.0
#     shift = jnp.pi / 2

#     def jvp_component(param_name: str) -> Array:
#         up_val = values.copy()
#         up_val[param_name] = up_val[param_name] + shift
#         f_up = no_shots_fwd(state, gates, observable, up_val)
#         down_val = values.copy()
#         down_val[param_name] = down_val[param_name] - shift
#         f_down = no_shots_fwd(state, gates, observable, down_val)
#         grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
#         return grad

#     val_keys = tuple(values.keys())
#     fwd = no_shots_fwd(state, gates, observable, values)
#     jvp_caller = jvp_component
#     if not isinstance(gates, Primitive):
#         param_to_gates: dict[str, tuple] = dict.fromkeys(val_keys, tuple())
#         for ind_gate, gate in enumerate(gates):
#             if is_parametric(gate) and gate.param in val_keys:  # type: ignore[attr-defined]
#                 param_to_gates[gate.param] += (ind_gate,)  # type: ignore[attr-defined]
#         print(param_to_gates)

#         if max(map(len, param_to_gates.values())) > 1:

#             def jvp_component_repeated_param(param_name: str) -> Array:
#                 shift_gates_ind = param_to_gates[param_name]

#                 def shift_jvp(ind_gate: int) -> Array:
#                     gate_shift = shift_gate(gates[ind_gate], shift)
#                     f_up = no_shots_fwd(
#                         state,
#                         gates[:ind_gate] + [gate_shift] + gates[min(ind_gate + 1, len(gates)) :],
#                         observable,
#                         values,
#                     )
#                     gate_shift = shift_gate(gates[ind_gate], -shift)
#                     f_down = no_shots_fwd(
#                         state,
#                         gates[:ind_gate] + [gate_shift] + gates[min(ind_gate + 1, len(gates)) :],
#                         observable,
#                         values,
#                     )
#                     grad = (
#                         spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
#                     )
#                     return grad

#                 return sum(shift_jvp(ind_gate) for ind_gate in shift_gates_ind)

#             jvp_caller = jvp_component_repeated_param
#     jvp = sum(jvp_caller(param) * tangent_dict[param] for param in val_keys)
#     return fwd, jvp.reshape(fwd.shape)


@no_shots_fwd.defjvp
def no_shots_fwd_jvp(
    state: Array,
    gates: Union[Primitive, Iterable[Primitive]],
    observable: list[Observable],
    primals: tuple[dict[str, float]],
    tangents: tuple[dict[str, float]],
) -> Array:
    values = primals[0]
    tangent_dict = tangents[0]

    # TODO: compute spectral gap through the generator which is associated with
    # a param name.
    spectral_gap = 2.0
    shift = jnp.pi / 2

    def jvp_component(param_name: str) -> Array:
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = no_shots_fwd(state, gates, observable, up_val)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = no_shots_fwd(state, gates, observable, down_val)
        grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
        return grad

    val_keys = tuple(values.keys())
    fwd = no_shots_fwd(state, gates, observable, values)
    jvp_caller = jvp_component
    if not isinstance(gates, Primitive):
        gate_names = list(
            map(
                lambda g: g.param if hasattr(g, "param") and isinstance(g.param, str) else None,
                gates,
            )
        )
        gate_names = list(filter(partial(is_not, None), gate_names))
        if len(gate_names) > len(val_keys):
            # raise NotImplementedError("Repeated params not possible")
            zero = jnp.zeros_like(fwd)

            def jvp_component_gate(index: int) -> Array:
                if not isinstance(gates[index], Parametric):
                    return zero
                if not isinstance(gates[index].param, str):
                    return zero
                altered_gate = shift_gate(gates[index], shift)
                f_up = no_shots_fwd(
                    state,
                    gates[:index] + [altered_gate] + gates[min(index + 1, len(gates)) :],
                    observable,
                    values,
                )
                altered_gate = shift_gate(gates[index], -shift)
                f_down = no_shots_fwd(
                    state,
                    gates[:index] + [altered_gate] + gates[min(index + 1, len(gates)) :],
                    observable,
                    values,
                )
                grad = spectral_gap * (f_up - f_down) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                return grad * tangent_dict[altered_gate.param]

            jvp = sum(jvp_component_gate(ind) for ind in range(len(gates)))
            return fwd, jvp.reshape(fwd.shape)
    jvp = sum(jvp_caller(param) * tangent_dict[param] for param in val_keys)
    return fwd, jvp.reshape(fwd.shape)


def shift_gate(gate: Parametric, shift_value: Array) -> Parametric:
    children, aux_data = gate.tree_flatten()
    return Parametric.tree_unflatten(aux_data[:-1] + (shift_value,), children)
