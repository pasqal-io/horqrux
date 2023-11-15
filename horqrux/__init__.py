from __future__ import annotations

from jax import config

config.update("jax_enable_x64", True)  # you should really really do this
