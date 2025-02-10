[![Linting / Tests/ Documentation](https://github.com/pasqal-io/horqrux/actions/workflows/run-tests-and-mypy.yml/badge.svg)](https://github.com/pasqal-io/horqrux/actions/workflows/run-tests-and-mypy.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pypi](https://badge.fury.io/py/horqrux.svg)](https://pypi.org/project/horqrux/)
![Coverage](https://img.shields.io/codecov/c/github/pasqal-io/horqrux?style=flat-square)

`horqrux` is a [JAX](https://jax.readthedocs.io/en/latest/)-based state vector and density matrix simulator designed for quantum machine learning and acts as a backend for [`Qadence`](https://github.com/pasqal-io/qadence), a digital-analog quantum programming interface.

## Installation

To install the CPU-only version, simply use `pip`:
```bash
pip install horqrux
```
If you intend to use GPU:

```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Getting started
`horqrux` adopts a minimalistic and functional interface however the [docs](https://pasqal-io.github.io/horqrux/latest/) provide a comprehensive A-Z guide ranging from how to apply simple primitive and parametric gates, to using [adjoint differentiation](https://arxiv.org/abs/2009.02823) to fit a nonlinear function and implementing [DQC](https://arxiv.org/abs/2011.10395) to solve a partial differential equation.

## Contributing

To learn how to contribute, please visit the [CONTRIBUTING](docs/CONTRIBUTING.md) page.

When developing within `horqrux`, you can either use the python environment manager [`hatch`](https://hatch.pypa.io/latest/):

```bash
pip install hatch

# enter a shell with containing all the dependencies
hatch shell

# run a command within the virtual environment with all the dependencies
hatch run python my_script.py
```

When using any other environment manager like `venv` or `conda`, simply do:

```bash
# within the virtual environment
pip install -e .
```
