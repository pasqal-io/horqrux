# HorQrux
A Jax-based state vector simulator.

## Installation

We've included a pyproject.toml for a CPU-only version, simply run
```
hatch shell
```
and you're ready to go. If you want to install the GPU version, the easiest is to open the shell
```
poetry shell
```
and run
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
