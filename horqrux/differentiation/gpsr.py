from __future__ import annotations

from functools import partial, singledispatch
from operator import is_not
from typing import Any, Iterable, Union

import jax
import jax.numpy as jnp
from jax import Array, random
from jax.experimental import checkify

from horqrux.apply import apply_gates
from horqrux.composite import Observable
from horqrux.differentiation.ad import _ad_expectation_single_observable
from horqrux.primitives import Parametric, Primitive
from horqrux.utils.operator_utils import DensityMatrix, State, expand_operator, num_qubits
from horqrux.utils.sparse_utils import stack_sp


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
    """Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectations analytically.

    Args:
        state (State): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.

    Returns:
        Array: Expectation values.
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
    return stack_sp(outputs)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 4, 5))
def finite_shots_fwd(
    state: State,
    gates: Union[Primitive, Iterable[Primitive]],
    observables: list[Observable],
    values: dict[str, float],
    n_shots: int = 100,
    key: Any = jax.random.PRNGKey(0),
) -> Array:
    """Run 'state' through a sequence of 'gates' given parameters 'values'
    and compute the expectations using `n_shots` shots per observable.

    Args:
        state (State): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): Sequence of gates.
        observables (list[Observable]): List of observables.
        values (dict[str, float]): Parameter values.
        n_shots (int, optional): Number of shots. Defaults to 100.
        key (Any, optional): Key for randomness. Defaults to jax.random.PRNGKey(0).

    Returns:
        Array: Expectation values.
    """
    output_gates = apply_gates(state, gates, values)
    n_qubits = num_qubits(output_gates)
    if isinstance(state, DensityMatrix):
        d = 2**n_qubits
        output_gates.array = output_gates.array.reshape((d, d))
    return eigen_sample(output_gates, observables, values, n_qubits, n_shots, key)


@jax.vmap
def validate_permutation_matrix(P: Array) -> Array:
    rows = P.sum(axis=0)
    columns = P.sum(axis=1)
    ones = jnp.ones(P.shape[0], dtype=rows.dtype)
    return ((ones == rows) & (ones == columns)).min()


def checkify_valid_permutation(P: Array) -> None:
    checkify.check(
        jnp.all(validate_permutation_matrix(P)),
        "Did not calculate valid permutation matrix",
    )


def align_eigenvectors(eigenvalues: Array, eigenvectors: Array) -> tuple[Array, Array]:
    """
    Given a list of eigenvalue eigenvector matrix tuples in the form of
    [(eigenvalue, eigenvector)...], this function aligns all the eigenvector
    matrices so that they are identical, and also rearranges the corresponding
    eigenvalues.

    This is primarily used as a utility function to help sample multiple
    correlated observables when using finite shots.
    """
    eigenvector = eigenvectors[0]

    P = jax.vmap(lambda mat: permutation_matrix(mat, eigenvector))(eigenvectors)
    checkify.checkify(checkify_valid_permutation)(P)
    aligned_eigenvalues = jax.vmap(jnp.dot)(eigenvalues, P).T
    return eigenvector, aligned_eigenvalues


def permutation_matrix(mat: Array, eigenvector_matrix: Array) -> Array:
    """Obtain the permutation matrix for aligning eigenvectors.

    Given two permuted eigenvector matrices, A and B, we wish to find a permutation
    matrix P such that A P = B. This function calculates such a permutation
    matrix and uses it to align each eigenvector matrix to the first eigenvector
    matrix of eigs.

    Args:
        mat (Array): Matrix A.
        eigenvector_matrix (Array): Eigenvector matrix B.

    Returns:
        Array: Permutation matrix P.
    """
    return (jnp.linalg.inv(mat) @ eigenvector_matrix).real > 0.5


@finite_shots_fwd.defjvp
def finite_shots_jvp(
    state: Array,
    gates: Union[Primitive, Iterable[Primitive]],
    observable: list[Observable],
    n_shots: int,
    key: Array,
    primals: tuple[dict[str, Array]],
    tangents: tuple[dict[str, Array]],
) -> tuple[Array, Array]:
    """Jvp version for finite_shots_fwd.

    Args:
        state (Array): Input state or density matrix.
        gates (Union[Primitive, Iterable[Primitive]]): sequence of gates.
        observable (list[Observable]): List of observables.
        n_shots (int): Number of shots.
        key (Array): Key for randomness.
        primals (tuple[dict[str, Array]]): Values we are differentiating over.
            Jax-specific syntax argument.
        tangents (tuple[dict[str, Array]]): Bases for differentiation.
            Jax-specific syntax argument.

    Returns:
        tuple[Array, Array]: Forward eveluation and gradient
    """
    values = primals[0]
    tangent_dict = tangents[0]

    val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)

    def jvp_component(param_name: str, key: Array) -> Array:
        up_key, down_key = random.split(key)
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = finite_shots_fwd(state, gates, observable, up_val, n_shots, up_key)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = finite_shots_fwd(state, gates, observable, down_val, n_shots, down_key)
        grad = (
            spectral_gap[param_name]
            * (f_up - f_down)
            / (4.0 * jnp.sin(spectral_gap[param_name] * shift / 2.0))
        )
        return grad

    params_with_keys = zip(values.keys(), random.split(key, len(values)))
    fwd = finite_shots_fwd(state, gates, observable, values, n_shots, key)
    jvp_caller = jvp_component
    if not isinstance(gates, Primitive):
        gate_names = extract_gate_names(gates)
        if len(gate_names) > len(val_keys):
            param_to_gates_indices = prepare_param_gates_seq(val_keys, gates)
            # repeated case
            if max(map(len, param_to_gates_indices.values())) > 1:  # type: ignore[arg-type]

                def jvp_component_repeated_param(param_name: str, key: Array) -> Array:
                    shift_gates = param_to_gates_indices[param_name]
                    shift_keys = random.split(key, len(shift_gates))

                    def shift_jvp(ind: int, key: Array) -> Array:
                        up_key, down_key = random.split(key)
                        spectral_gap = gates[ind].spectral_gap  # type: ignore[index]
                        gates_up = alter_gate_sequence(gates, ind, shift)
                        f_up = finite_shots_fwd(
                            state, gates_up, observable, values, n_shots, up_key
                        )
                        gates_down = alter_gate_sequence(gates, ind, -shift)
                        f_down = finite_shots_fwd(
                            state, gates_down, observable, values, n_shots, down_key
                        )
                        return (
                            spectral_gap
                            * (f_up - f_down)
                            / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                        )

                    return sum(
                        shift_jvp(shift_ind, key) for shift_ind, key in zip(shift_gates, shift_keys)
                    )

            else:
                spectral_gap = spectral_gap = spectral_gap_from_gates(
                    param_to_gates_indices, val_keys
                )

            jvp_caller = jvp_component_repeated_param

    jvp = sum(jvp_caller(param, key) * tangent_dict[param] for param, key in params_with_keys)
    return fwd, jvp.reshape(fwd.shape)


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

    val_keys, spectral_gap, shift = initialize_gpsr_ingredients(values)

    def jvp_component(param_name: str) -> Array:
        up_val = values.copy()
        up_val[param_name] = up_val[param_name] + shift
        f_up = no_shots_fwd(state, gates, observable, up_val)
        down_val = values.copy()
        down_val[param_name] = down_val[param_name] - shift
        f_down = no_shots_fwd(state, gates, observable, down_val)
        grad = (
            spectral_gap[param_name]
            * (f_up - f_down)
            / (4.0 * jnp.sin(spectral_gap[param_name] * shift / 2.0))
        )
        return grad

    fwd = no_shots_fwd(state, gates, observable, values)
    jvp_caller = jvp_component
    if not isinstance(gates, Primitive):
        gate_names = extract_gate_names(gates)
        if len(gate_names) > len(val_keys):
            param_to_gates_indices = prepare_param_gates_seq(val_keys, gates)
            # repeated case
            if max(map(len, param_to_gates_indices.values())) > 1:  # type: ignore[arg-type]

                def jvp_component_repeated_param(param_name: str) -> Array:
                    shift_gates = param_to_gates_indices[param_name]

                    def shift_jvp(ind: int) -> Array:
                        spectral_gap = gates[ind].spectral_gap  # type: ignore[index]
                        gates_up = alter_gate_sequence(gates, ind, shift)
                        f_up = no_shots_fwd(state, gates_up, observable, values)
                        gates_down = alter_gate_sequence(gates, ind, -shift)
                        f_down = no_shots_fwd(state, gates_down, observable, values)
                        return (
                            spectral_gap
                            * (f_up - f_down)
                            / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                        )

                    return sum(shift_jvp(shift_ind) for shift_ind in shift_gates)

                jvp_caller = jvp_component_repeated_param
            else:
                spectral_gap = spectral_gap_from_gates(param_to_gates_indices, val_keys)

    jvp = sum(jvp_caller(param) * tangent_dict[param] for param in val_keys)
    return fwd, jvp.reshape(fwd.shape)


def to_shift(gate: Parametric, shift_value: float) -> Any:
    """Create the shifted gate for PSR.

    Args:
        gate (Parametric): Gate to shift.
        shift_value (float): Shift value.

    Returns:
        Any: A new Parametric (Any type due to clsmethod)
    """
    children, aux_data = gate.tree_flatten()
    return Parametric.tree_unflatten(aux_data[:-1] + (aux_data[-1] + shift_value,), children)


def prepare_param_gates_seq(
    param_names: tuple[str, ...], gates: Iterable[Primitive]
) -> dict[str, tuple]:
    """Create a dictionary of parameter names and corresponding parametric gates.

    Args:
        param_names (tuple[str, ...]): Parameters.
        gates (Iterable[Primitive]): Sequence of gates.

    Returns:
        dict[str, tuple]: dictionary of parameter names and corresponding parametric gates.
    """
    param_to_gates: dict[str, tuple] = dict.fromkeys(param_names, tuple())
    for i, gate in enumerate(gates):
        if gate.is_parametric and gate.param in param_names:  # type: ignore[attr-defined]
            param_to_gates[gate.param] += (i,)  # type: ignore[attr-defined]
    return param_to_gates


def alter_gate_sequence(gates: Iterable[Any], ind_alter: int, shift_val: float) -> Any:
    """Create a sequence replacing the `ind_alter` gate by its shifted version.

    Args:
        gates (Iterable[Any]): sequence of gates.
        ind_alter (int): Index of gate to shift.
        shift_val (float): Shift value.

    Returns:
        Any: sequence of gates including shift.
    """
    gate_alter = gates[ind_alter]  # type: ignore[index]
    gate_shift = to_shift(gate_alter, shift_val)
    upper = min(ind_alter + 1, len(gates))  # type: ignore[arg-type]
    gates_seq = gates[:ind_alter] + [gate_shift] + gates[upper:]  # type: ignore[index]
    return gates_seq


def initialize_gpsr_ingredients(
    values: dict[str, float]
) -> tuple[tuple[str, ...], dict[str, float], Array]:
    """Initialize the parameter names, spectral_gap, and shift value for GPSR.

    Args:
        values (dict[str, float]): Parameter values.

    Returns:
        tuple[tuple[str, ...], dict[str, float], Array]: parameter names, spectral_gap, and shift value.
    """
    val_keys = tuple(values.keys())
    spectral_gap = dict.fromkeys(val_keys, 2.0)
    shift = jnp.pi / 2
    return val_keys, spectral_gap, shift


def extract_gate_names(gates: Iterable[Primitive]) -> list:
    """Extract gate names when a gate is parametric.

    Args:
        gates (Iterable[Primitive]): List of gates.

    Returns:
        list: list of names.
    """
    gate_names = list(
        map(
            lambda g: g.param if hasattr(g, "param") and isinstance(g.param, str) else None,
            gates,
        )
    )
    gate_names = list(filter(partial(is_not, None), gate_names))
    return gate_names


def spectral_gap_from_gates(
    param_to_gates_indices: dict[str, tuple], param_names: Iterable[str]
) -> dict[str, Array]:
    """Extract spectral gap from each gate.

    Only works when a parameter name is used by only one gate.

    Args:
        param_to_gates_indices (dict[str, tuple]): dictionary mapping
            parameter name to indices of gates.
        param_names (Iterable[str]): Name of parameters.

    Returns:
        dict[str, Array]: Parameter names mapped with the spectral gap.
    """
    return {param: param_to_gates_indices[param][0].spectral_gap for param in param_names}
