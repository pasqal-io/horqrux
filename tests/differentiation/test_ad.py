from __future__ import annotations

import chex
import jax
from jax import Array

from horqrux import expectation, random_state
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives.parametric import RX, RZ
from tests.utils import verify_arrays


class GPSRTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_ad_obs(self) -> None:
        param_name = "theta"
        x = jax.random.uniform(jax.random.key(0), (3,))
        param_names = [param_name, param_name + "2", param_name + "obs"]
        ops = [RX(param_names[0], 0), RX(param_names[1], 1)]

        def values_to_dict(x, y=None) -> dict[str, Array]:
            values_circuit = {param_names[0]: x[0], param_names[1]: x[1], param_names[2]: x[2]}
            if y is None:
                return values_circuit
            else:
                return {"circuit": values_circuit, "observables": {param_names[2]: y}}

        circuit = QuantumCircuit(2, ops)
        observables = [Observable([RZ(param_name + "obs", 0)])]
        state = random_state(2)

        @self.variant
        def exp_fn(x, y=None) -> Array:
            values = values_to_dict(x, y)
            return expectation(state, circuit, observables, values, diff_mode="ad")

        d_exact = jax.grad(lambda x: exp_fn(x).sum())(x)

        d_exact_circuit_params = jax.grad(lambda x, y: exp_fn(x, y).sum(), argnums=0)(x[:2], x[-1])
        d_exact_obs_params = jax.grad(lambda x, y: exp_fn(x, y).sum(), argnums=1)(x[:2], x[-1])

        assert verify_arrays(d_exact[:-1], d_exact_circuit_params)
        assert verify_arrays(d_exact[-1], d_exact_obs_params)
