# wildfires

[![License: MIT](https://img.shields.io/badge/License-MIT-blueviolet)](https://github.com/akuhnregnier/wildfires/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Build Status](https://travis-ci.com/akuhnregnier/wildfires.svg?branch=master)](https://travis-ci.com/akuhnregnier/wildfires)
[![codecov](https://codecov.io/gh/akuhnregnier/wildfires/branch/master/graph/badge.svg)](https://codecov.io/gh/akuhnregnier/wildfires)

## Install

wildfires requires Python >= 3.7.

It needs to be installed from source (note the use of SSH instead of HTTPS), either using:
```sh
pip install git+ssh://git@github.com/akuhnregnier/wildfires.git@master#egg=wildfires
```
or after cloning (forking) for development purposes:
```sh
git clone git@github.com:akuhnregnier/wildfires.git
pip install -e "wildfires"
```

Required packages are not installed using those commands, because the `requirements.yml` environment file was designed to facilitate package installation using `conda` (see [here](#installing-dependencies) or [here](#creating-a-new-conda-environment-manually)).

JupyterLab ipywidgets:
```sh
conda install ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Development

The instructions below assume that you are using Linux and managing Python using [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

**All** essential configuration steps detailed in the section [Prerequisites](#prerequisites) are automatically carried out by `dev/bootstrap.py`.
To view options for this initial setup, run
```sh
dev/bootstrap.py --help
```

#### Installing JupyterLab extensions

```sh
dev/install_jupyter_extensions.py
```

## Prerequisites

### Installing Dependencies

This information is mainly for reference and additional documentation of `dev/bootstrap.py`, or if individual parts (like using `pytest`) should be carried out independently.


#### Installing Into an Existing Environment

Installing all packages and configuring all tools:
```sh
dev/bootstrap.py existing
```

#### Installing Into a New Environment

By default, the new environment will be called 'wildfires'.
Installation will fail if an environment with this name is already present, unless the `--force` flag is supplied, as shown below:
```sh
dev/bootstrap.py new --force
```

Internally `dev/bootstrap.py new` uses `conda env create` with support for the `--name` and `--force` options.

#### Creating a new Conda Environment Manually

Creating a new environment called 'wildfires':
```sh
conda env create --file requirements.yaml --name wildfires
```

Overwriting an existing environment called 'wildfires':
```sh
conda env create --file requirements.yaml --name wildfires --force
```

#### Using a New Environment

Making a new environment the local default environment for the repository using pyenv (assuming `--name wildfires` was used):
```sh
pyenv activate wildfires
pyenv local $PYENV_VERSION
```

### Hooks and Tests

To set up pre-commit hooks, run
```sh
pre-commit install
```
This needs to be done **every time** the repository is cloned (as is done by `dev/bootstrap.py`)!

#### Manual pre-commit Testing

```sh
pre-commit run --verbose --all-files
```

#### Pytest

Run `pytest` in the root directory of the repository, so that all tests will be discovered.
If DeprecationWarning warnings are desired, run `pytest -Wd` instead.

Note that pytest is also run as part of the pre-commit hooks set up by pre-commit.
