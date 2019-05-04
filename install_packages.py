#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install conda and pip packages into the currently active environment.

This utilises a requirements.yml file generated using the command:
    `conda env export > requirements.yml`

As per the guidelines at https://www.anaconda.com/using-pip-in-a-conda-environment/
pip will be run after conda.

"""

if __name__ == "__main__":
    from subprocess import run

    run(("conda", "install", "--yes", "pyyaml"))

    import yaml

    with open("requirements.yml", "r") as f:
        try:
            requirements = yaml.safe_load(f)
        except yaml.YAMLError as exception:
            print(exception)

    conda_dependencies = [
        dependency
        for dependency in requirements["dependencies"]
        if isinstance(dependency, str)
    ]
    assert (
        len(conda_dependencies) == len(requirements["dependencies"]) - 1
    ), "Only 1 entry (the other dependency dict) should be missing."

    other_dependency_dict = [
        dependency
        for dependency in requirements["dependencies"]
        if isinstance(dependency, dict)
    ][0]
    assert len(other_dependency_dict.keys()) == 1, "Only expecting 'pip' entry"
    pip_dependencies = other_dependency_dict["pip"]

    if run(("conda", "install", "--yes", *conda_dependencies)).returncode:
        raise RuntimeError("Conda install failed.")

    if run(("pip", "install", *pip_dependencies)).returncode:
        raise RuntimeError("Pip install failed.")

    if run(
        ("jupyter", "labextension", "install", "@ryantam626/jupyterlab_code_formatter")
    ).returncode:
        raise RuntimeError("jupyterlab_code_formatter installation failed.")

    if run(
        ("jupyter", "serverextension", "enable", "--py", "jupyterlab_code_formatter")
    ).returncode:
        raise RuntimeError("jupyterlab_code_formatter activation failed.")

    if run(
        ("jupyter", "nbextension", "enable", "nbdime", "--py", "--sys-prefix")
    ).returncode:
        raise RuntimeError("nbdime enabling failed.")
