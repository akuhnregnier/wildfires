#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install dependencies and configure tools.

Pre-commit and nbstripout will be configured by default.

Install conda and pip packages by creating a new environment or by installing the
packages into the existing environment.

This utilises a requirements.yaml file generated using the command:
`conda env export > requirements.yaml`

As per the guidelines at https://www.anaconda.com/using-pip-in-a-conda-environment/
pip will be run after conda (only relevant if installing into an existing
environment).

"""
import argparse
import os
from subprocess import check_output, run

YAML_REQUIREMENTS = "requirements.yaml"
DEFAULT_ENV_NAME = "wildfires"


def install_into_existing_env():
    """Install dependencies from the file 'yaml_requirements'."""
    try:
        import yaml
    except ModuleNotFoundError:
        print("yaml not found, installing 'pyyaml' using conda.")
        run(("conda", "install", "--yes", "pyyaml"))
        import yaml

    with open(YAML_REQUIREMENTS, "r") as f:
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
        raise RuntimeError("conda dependency installation failed.")

    if run(("pip", "install", *pip_dependencies)).returncode:
        raise RuntimeError("pip dependency installation failed.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Install dependencies and configure tools. Pre-commit and nbstripout "
            "will be configured by default."
        )
    )
    subparsers = parser.add_subparsers(required=True, dest="command")

    new_env = subparsers.add_parser(
        "new",
        description=(
            "Create a new environment. The new environment can be made the local "
            "default environment using `pyenv local $PYENV_VERSION` after running "
            "`pyenv activate NAME`"
        ),
        help="Create a new environment",
    )
    new_env.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing environment of the same name",
    )
    new_env.add_argument(
        "--name",
        help="Name of new environment (default: %(default)s)",
        default=DEFAULT_ENV_NAME,
    )

    existing_help = "Install dependencies into currently active environment"
    subparsers.add_parser("existing", description=existing_help, help=existing_help)

    config_group = parser.add_argument_group("configuration")
    config_group.add_argument(
        "--skip-all", action="store_true", help="Do not carry out any configuration"
    )
    config_group.add_argument(
        "--skip-nbstripout", action="store_true", help="Do not install nbstripout"
    )
    config_group.add_argument(
        "--skip-pre-commit", action="store_true", help="Do not install pre-commit"
    )

    args = parser.parse_args()

    if args.command == "existing":
        install_into_existing_env()
    elif args.command == "new":
        new_env_cmd = [
            "conda",
            "env",
            "create",
            "--file",
            YAML_REQUIREMENTS,
            "--name",
            args.name,
        ]
        if args.force:
            new_env_cmd.append("--force")

        if run(new_env_cmd).returncode:
            raise RuntimeError(
                "Installation into new environment failed. Use `./bootstrap.py new "
                "--force` to overwrite an existing environment."
            )

        # Filter out versions that match the new environment name.
        possible_versions = list(
            filter(
                lambda name: args.name in name,
                map(
                    lambda entry: entry.strip().strip("* ").split(" ")[0],
                    check_output(("pyenv", "versions")).decode().split("\n"),
                ),
            )
        )
        assert (
            len(possible_versions) > 0
        ), "There should be 1 (or more) matching version(s)."

        # Activate the new environment using the 'PYENV_VERSION' environment variable.
        new_pyenv_version = max(possible_versions, key=lambda s: len(s))
        print(f"Activating internally: {new_pyenv_version}")
        os.environ["PYENV_VERSION"] = new_pyenv_version

        # possible_versions[0] should be the same as $PYENV_VERSION by this point.
        env_activate_instructions = (
            "To activate the new environment and make it the local default using "
            "pyenv, run the following commands:\n\n"
            "\tpyenv activate {}\n"
            "\tpyenv local $PYENV_VERSION\n"
        ).format(args.name)
        print(env_activate_instructions)
    else:
        raise ValueError("Unknown command '{}'.".format(args.command))

    if args.skip_all:
        return
    if not args.skip_nbstripout:
        code_sum = 0
        code_sum += run(
            ("git", "config", "filter.nbstripout.clean", "nbstripout-fast")
        ).returncode
        code_sum += run(("git", "config", "filter.nbstripout.smudge", "cat")).returncode
        code_sum += run(
            ("git", "config", "filter.nbstripout.required", "true")
        ).returncode
        code_sum += run(
            ("git", "config", "diff.ipynb.textconv", "nbstripout-fast -t")
        ).returncode
    if not args.skip_pre_commit:
        if run(("pre-commit", "install")).returncode:
            raise RuntimeError("pre-commit installation failed.")


if __name__ == "__main__":
    main()
