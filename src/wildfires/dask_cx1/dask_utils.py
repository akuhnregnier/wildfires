# -*- coding: utf-8 -*-
import math
import shlex
from subprocess import Popen, check_call, check_output
from time import monotonic, sleep

from dask.distributed import get_worker

from ..ports import get_ports
from .dask_cx1 import get_client, strip_multiline


class NoNotebookError(RuntimeError):
    """Raised when a matching notebook could not be found within the timeout."""


def get_running_procs(client=None, user="ahk114", workers=None):
    """Return a string containing `user`'s currently running processes.

    Args:
        client (dask.distributed.Client): Client used to submit functions. If None,
            `wildfires.dask_cx1.get_client()` will be used to attempt automatic
            retrieval of an existing cluster.
        user (str): User for which to query running processes.
        workers (str): If given, will be used to define on which worker(s) the
            function should be run.

    Returns:
        str: The running processes.

    """
    if client is None:
        client = get_client()

    def _get_running_procs():
        return check_output(shlex.split(f"pgrep -afu {user}")).decode()

    return client.submit(_get_running_procs, workers=workers, pure=False).result()


def start_jupyter_lab(
    client=None,
    workers=None,
    jupyter_exc="/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/jupyter",
    notebook_dir="~/Documents/PhD/wildfire-analysis/analyses/",
    target_user="alexander",
    target_host="maritimus.webredirect.org",
):
    """Set up a JupyterLab environment on a specified worker.

    This is followed by port forwarding to the target host.

    Args:
        client (dask.distributed.Client): Client used to submit functions. If None,
            `wildfires.dask_cx1.get_client()` will be used to attempt automatic
            retrieval of an existing cluster.
        worker (str): This is used to set the worker the function should be run on.
        jupyter_exc (str): Path to the Jupyter executable to use.
        notebook_dir (str): The working directory.
        target_user (str): SSH username.
        target_host (str): SSH hostname.

    Returns:
        int: The port used by the JupyterLab environment.
        address: The JupyterLab worker address.

    """
    if client is None:
        client = get_client()

    def _start_jupyter_lab():
        port = get_ports()[0]
        # This will keep running the background to serve the JupyterLab environment.
        Popen(
            shlex.split(
                strip_multiline(
                    f"""{jupyter_exc} lab --no-browser --port={port}
                    --notebook-dir={notebook_dir}"""
                )
            )
        )
        # If this succeeds, it will automatically go to the background.
        # NOTE: There is a (small) chance that the chosen port is not available on the
        # target host, which is not dealt with here.
        check_call(
            shlex.split(
                f"ssh -f -NT -R {port}:localhost:{port} {target_user}@{target_host}"
            )
        )
        return port, get_worker().address

    return client.submit(_start_jupyter_lab, workers=workers, pure=False).result()


def list_notebooks(
    client=None,
    jupyter_exc="/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/jupyter",
    filter_port=None,
    workers=None,
    timeout=10,
):
    """List notebooks currently running off of the Jupyter executable `jupyter_exc`.

    Args:
        client (dask.distributed.Client): Client used to submit functions. If None,
            `wildfires.dask_cx1.get_client()` will be used to attempt automatic
            retrieval of an existing cluster.
        jupyter_exc (str): Path to the Jupyter executable to use.
        filter_port (int): If given, only notebooks running on 'localhost:`port`' will
            be listed.
        workers (str): If given, will be used to define which worker(s) the function
            should be run on.
        timeout (float or None): Timeout before we stop looking for a matching
            notebook. If `timeout` is None, the timeout is disabled.

    Returns:
        str: The running notebook(s) and the associated Dask worker address.

    Raises:
        NoNotebookError: If no matching notebook could not be found within the timeout.

    """
    if client is None:
        client = get_client()

    def _list_notebooks():
        notebooks = (
            check_output(shlex.split(f"{jupyter_exc} notebook list"))
            .decode()
            .split("\n")
        )
        if filter_port is not None:
            notebooks = [
                notebook
                for notebook in notebooks
                if f"localhost:{filter_port}" in notebook
            ]

        if notebooks:
            return "\n".join([f"Running on worker: {get_worker().address}", *notebooks])

    if timeout is None:
        timeout = math.inf

    start = monotonic()
    while (monotonic() - start) < timeout:
        notebook_listing = client.submit(
            _list_notebooks, workers=workers, pure=False
        ).result()
        if notebook_listing is None:
            sleep(0.5)
            continue
        return notebook_listing
    else:
        # We have timed out without finding a matching notebook.
        raise NoNotebookError(
            f"No matching notebook was found within the timeout ({timeout} s)."
        )


def setup_remote_jupyter_lab(
    client=None,
    workers=None,
    jupyter_exc="/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/jupyter",
    notebook_dir="~/Documents/PhD/wildfire-analysis/analyses/",
    target_user="alexander",
    target_host="maritimus.webredirect.org",
):
    """Set up a JupyterLab environment on a specified worker.

    This is followed by port forwarding to the target host.

    Here, `start_jupyter_lab()` and `list_notebooks()` are combined to start the
    JupyterLab environment and then retrieve its address and token.

    Args:
        client (dask.distributed.Client): Client used to submit functions. If None,
            `wildfires.dask_cx1.get_client()` will be used to attempt automatic
            retrieval of an existing cluster.
        workers (str): This is used to set the worker the function should be run on.
        jupyter_exc (str): Path to the Jupyter executable to use.
        notebook_dir (str): The working directory.
        target_user (str): SSH username.
        target_host (str): SSH hostname.

    Returns:
        str: The running notebook(s) including their access tokens and the Dask worker
            address.

    """
    if client is None:
        client = get_client()

    port, worker_address = start_jupyter_lab(
        client, workers, jupyter_exc, notebook_dir, target_user, target_host
    )
    if workers is None:
        # If no worker was explicitly requested, make sure that the notebooks are
        # listed on the worker JupyterLab is running on.
        workers = worker_address
    return list_notebooks(client, jupyter_exc, port, workers)
