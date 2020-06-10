# -*- coding: utf-8 -*-
import math
import os
import shlex
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import Popen, check_call, check_output
from tempfile import NamedTemporaryFile
from time import monotonic, sleep

from dask.distributed import get_worker

from ..ports import get_ports
from .dask_cx1 import get_client, strip_multiline


class NoNotebookError(RuntimeError):
    """Raised when a matching notebook could not be found within the timeout."""


def common_worker_threads(client):
    """Get the common value of the 'threads' resource.

    Args:
        client (`dask.distributed.Client`): Dask Client used to submit tasks to the
            scheduler.

    Returns:
        int: Common value of the 'threads' resource across the client's workers.

    Raises:
        RuntimeError: If the Dask workers associated with `client` do not specify the 'threads'
            resource.
        RuntimeError: If the 'threads' resources specified by the Dask workers associated with
            `client` do not match.

    """
    all_worker_resources = [
        worker["resources"] for worker in client.scheduler_info()["workers"].values()
    ]
    if not all("threads" in resources for resources in all_worker_resources):
        raise RuntimeError(
            "Expected all workers to specify the 'threads' resource, but got "
            f"{all_worker_resources}."
        )

    all_worker_threads = [resource["threads"] for resource in all_worker_resources]
    if not all(threads == all_worker_threads[0] for threads in all_worker_threads):
        raise RuntimeError(
            "Expected all workers to have the same number of threads, but got "
            f"{all_worker_threads}."
        )
    return int(all_worker_threads[0])


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
        target_user (str): SSH username (eg. `getpass.getuser()`).
        target_host (str): SSH hostname (eg. `socket.getfqdn()`).

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
                "-o StrictHostKeyChecking=no"
            )
        )
        return port, get_worker().address

    return client.submit(_start_jupyter_lab, workers=workers, pure=False).result()


def list_notebooks(
    client=None,
    jupyter_exc="/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/jupyter",
    filter_port=None,
    workers=None,
    timeout=60,
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
        target_user (str): SSH username (eg. `getpass.getuser()`).
        target_host (str): SSH hostname (eg. `socket.getfqdn()`).

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


def start_general_lab(
    jupyter_exc="/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/jupyter",
    notebook_dir="~/Documents/PhD/wildfire-analysis/analyses/",
    valid_ports_exec=(
        "/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/valid-ports"
    ),
    target_user="alexander",
    target_host="maritimus.webredirect.org",
    walltime="05:00:00",
    start_timeout=900,
    **cluster_kwargs,
):
    """Start a Dask cluster using workers of the general class.

    Start a JupyterLab environment on the (first) scheduler job by default.

    Args:
        jupyter_exc (str): Path to the Jupyter executable to use.
        notebook_dir (str): The working directory.
        jupyter_exc (str): Path to the Jupyter executable to use.
        valid_ports_exec (str): Path to the `valid-ports` executable.
        target_user (str): SSH username (eg. `getpass.getuser()`).
        target_host (str): SSH hostname (eg. `socket.getfqdn()`).
        walltime (str): Walltime limit, eg. '05:00:00'.
        start_timeout (float): Time in seconds to wait for the job to start in order
            to parse the created output files for the JupyterLab address.
        cluster_kwargs: Additional keyword arguments are passed to the
            `CX1GeneralCluster()` initialiser.

    """
    job_name = "JupyterLab-general"
    ssh_opts = "-f -NT -o StrictHostKeyChecking=no"
    job_script = f"""
#!/usr/bin/env bash

#PBS -N {job_name}
#PBS -l select=1:ncpus=32:mem=60GB
#PBS -l walltime={walltime}
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
#
JOBID="${{PBS_JOBID%%.*}}"
echo $(date): JOBID $JOBID on host $(hostname).
#
# Get a free port.
#
read JUPYTERPORT <<< $({valid_ports_exec} 1)
#
# Make the JupyterLab environment accessible from a user-facing node.
#
ssh -R $JUPYTERPORT:localhost:$JUPYTERPORT {target_user}@{target_host} {ssh_opts}
#
# Start JupyterLab using the determined port.
#
{jupyter_exc} lab --no-browser --port=$JUPYTERPORT --notebook-dir={notebook_dir}
"""

    with NamedTemporaryFile(prefix="jupyterlab_general_", suffix=".sh") as job_file:
        with open(job_file.name, "w") as f:
            f.write(job_script)
        job_str = check_output(shlex.split(f"qsub -V {job_file.name}")).decode()

    job_id = job_str.split(".")[0]
    output_file = ".e".join((job_name, job_id))
    print(
        strip_multiline(
            """Note: If the prompt 'Would you like to clear the workspace or keep
            waiting.' is encountered, try executing
            'rm -rf ~/.jupyter/lab/workspaces/' before clicking on the link below
            again."""
        )
    )
    print("Waiting for job to start...")
    start = monotonic()
    while (monotonic() - start) < start_timeout:
        sleep(5)
        if os.path.isfile(output_file):
            with open(output_file) as f:
                lines = [l for l in f.readlines() if "localhost" in l]
            if lines:
                address = "".join(lines)
                print(address)
                return address


def main():
    parser = ArgumentParser(
        description=(
            "Start a JupyterLab environment using a 'general' job class worker."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--jupyter-exc",
        default="/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/jupyter",
    )
    parser.add_argument(
        "--notebook-dir", default="~/Documents/PhD/wildfire-analysis/analyses/"
    )
    parser.add_argument(
        "--valid-ports-exec",
        default=(
            "/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/valid-ports"
        ),
    )
    parser.add_argument("--target-user", default="alexander")
    parser.add_argument("--target-host", default="maritimus.webredirect.org")
    parser.add_argument("--walltime", default="05:00:00")
    parser.add_argument("--start-timeout", default=900, type=float)

    args = parser.parse_args()

    start_general_lab(
        jupyter_exc=args.jupyter_exc,
        notebook_dir=args.notebook_dir,
        valid_ports_exec=args.valid_ports_exec,
        target_user=args.target_user,
        target_host=args.target_host,
        walltime=args.walltime,
        start_timeout=args.start_timeout,
    )
