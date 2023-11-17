# horqrux

**horqrux** is a [JAX](https://jax.readthedocs.io/en/latest/)-based state vector simulator designed for quantum machine learning.
It acts as a backend for [`Qadence`](https://github.com/pasqal-io/qadence), a digital-analog quantum programming interface.

## Installation

`horqrux` (CPU-only) can be installed from PyPI with `pip` as follows:
```bash
pip install horqrux
```
If you want to install the GPU version, simply do:

```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

[![Linting / Tests/ Documentation](https://github.com/pasqal-io/horqrux/actions/workflows/run-tests-and-mypy.yml/badge.svg)](https://github.com/pasqal-io/horqrux/actions/workflows/run-tests-and-mypy.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pypi](https://badge.fury.io/py/horqrux.svg)](https://pypi.org/project/horqrux/)


## Install from source

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `horqrux` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

Please note that `hatch` will not combine nicely with other environment managers such Conda. If you want to use Conda, install `horqrux` from source using `pip`:

```bash
# within the Conda environment
python -m pip install -e .
```

## Contributing

Please refer to [CONTRIBUTING](docs/CONTRIBUTING.md) to learn how to contribute to `horqrux`.
