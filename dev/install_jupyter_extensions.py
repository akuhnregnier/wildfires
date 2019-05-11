#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install JupyterLab extensions."""

import argparse
from subprocess import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Install JupyterLab extensions."))
    parser.parse_args()
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
