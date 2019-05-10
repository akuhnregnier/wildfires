# wildfires

## Development

### Installing Dependencies

#### Creating a new Conda Environment

Creating a new environment called 'wildfires' (the default specified in `requirements.yaml`):
```sh
conda env create --file requirements.yaml
```

Overwriting an existing environment of the same name:
```sh
conda env create --file requirements.yaml --force
```

Creating a new environment called 'foo':
```sh
conda env create --file requirements.yaml --name foo
```

#### Installing Into an Existing Environment

Installing all packages and configuring JupyterLab extensions:
```sh
./install_packages.py
```

Installing all packages without configuring JupyterLab extensions:
```sh
./install_packages.py --skip-jupyter
```

### Hooks and Tests

To set up pre-commit hooks, run
```sh
pre-commit install
```
This needs to be done **every time** the repository is cloned!

#### Manual pre-commit Testing

Run
```sh
pre-commit run --verbose --all-files
```

#### Pytest

Run `pytest` in the root directory of the repository, so that all tests will be discovered.
If DeprecationWarning warnings are desired, run `pytest -Wd` instead.

Note that pytest is also run as part of the pre-commit hooks set up by pre-commit.
