# -*- coding: utf-8 -*-
import json
import logging
import os
import re
import shlex
import socket
import sys
from functools import wraps
from inspect import signature
from operator import eq, ge
from random import choice
from string import ascii_lowercase
from subprocess import Popen
from textwrap import dedent

from dask.distributed import Client
from dask.utils import parse_bytes
from joblib import parallel_backend

from dask_jobqueue.pbs import PBSCluster, PBSJob, pbs_format_bytes_ceil

from .ports import get_ports
from .qstat import get_ncpus

logger = logging.getLogger(__name__)

SCHEDULER_DIR = os.path.expanduser(os.path.join("~", "schedulers"))

DEFAULTS = {"cores": 1, "memory": "1.1GiB", "walltime": "01:00:00"}

cluster_size_error_msg = (
    "Since forwarding of worker ports may fail, workers added after the initial "
    "timeout may not behave as expected."
)


class DaskCX1Error(Exception):
    """Base class for exceptions in the dask_cx1 module."""


class FoundSchedulerError(DaskCX1Error):
    """Raised when no scheduler matching the requested specifications could be found."""


class WorkerPortError(DaskCX1Error):
    """Raised when an error while determining worker port numbers."""


def walltime_seconds(walltime):
    """Given a walltime string, return the number of seconds it represents.

    Args:
        walltime (str): Walltime, eg. '01:00:00'.

    Returns:
        int: Number of seconds represented by the walltime.

    Raises:
        ValueError: If an unsupported walltime format is supplied.

    """
    match = re.fullmatch(r"(\d{1,3}):(\d{1,2}):(\d{1,2})", walltime)
    if match is None:
        raise ValueError(
            f"Expected walltime like '02:30:40', but got {repr(walltime)}."
        )
    hours, minutes, seconds = match.groups()
    return int(hours) * 60 * 60 + int(minutes) * 60 + int(seconds)


def get_client(**specs):
    """Try to connect to an existing Dask scheduler with the supplied specs.

    Args:
        specs: Worker resource specifications, eg. cores=10, memory="8GB". The
            scheduler's cluster will have access to a certain number of such workers,
            where each worker will have access to `cores` number of threads.

    Returns:
        distributed.client.Client: Client used to connect to the scheduler.

    Raises:
        FoundSchedulerError: If no matching scheduler could be found.

    """
    logger.info(f"Trying to find Dask scheduler with minimum specs {specs}.")
    scheduler_file = get_scheduler_file(**specs)
    if scheduler_file is not None:
        logger.info(f"Found scheduler at {scheduler_file}.")
        return Client(scheduler_file=scheduler_file)
    raise FoundSchedulerError(f"No scheduler with minimum specs {specs} found.")


def get_parallel_backend(loky_fallback=True, **specs):
    """Try to connect to an existing Dask scheduler with the supplied specs.

    The fallback is a local loky backend with the number of CPUs determined by the
    currently running job (if any, eg. a JupyterLab job), as reported
    `wildfires.utils.get_ncpus()`.

    Args:
        specs: Worker resource specifications, eg. cores=10, memory="8GB". The
            scheduler's cluster will have access to a certain number of such workers,
            where each worker will have access to `cores` number of threads.
        loky_fallback (bool): If True, allow falling back to a local loky cluster when
            no matching scheduler is found.

    Returns:
        joblib.parallel.parallel_backend: Context manager to manage nested calls to
        `joblib.parallel.Parallel()`.

    Raises:
        FoundSchedulerError: If no matching scheduler could be found and
            `loky_fallback` is False.

    """
    logger.info(f"Trying to find Dask scheduler with minimum specs {specs}.")
    scheduler_file = get_scheduler_file(**specs)
    if scheduler_file is not None:
        logger.info(f"Found scheduler at {scheduler_file}.")
        # TODO: Turn this function into a context manager of its own to handle Client
        # shutdown.
        Client(scheduler_file=scheduler_file)
        return parallel_backend("dask", wait_for_workers_timeout=600)

    if not loky_fallback:
        raise FoundSchedulerError(f"No scheduler with minimum specs {specs} found.")

    ncpus = get_ncpus()
    logger.info(f"Using loky backend with {ncpus} jobs.")
    return parallel_backend("loky", n_jobs=ncpus)


def get_scheduler_file(match="above", **specs):
    """Get a scheduler file that points to an existing scheduler.

    This file can then be used to connect to the scheduler like so:

    >>> from dask.distributed import Client  # doctest: +SKIP
    >>> scheduler_file = get_scheduler_file(cores=3, memory="10GB")  # doctest: +SKIP
    >>> client = Client(scheduler_file=scheduler_file)  # doctest: +SKIP

    This client can then be used to submit tasks to the scheduler.

    Args:
        match (str): One of "above" or "same". All resource specifications provided as
            keyword arguments (eg. 'cores=10') will be matched against the worker
            specifications in existing scheduler files. If "above", all specifications
            must match or exceed the requested values for a match to be reported. If
            "same", all specifications must match exactly.
        specs: Worker resource specifications, eg. cores=10, memory="8GB". The
            scheduler's cluster will have access to a certain number of such workers,
            where each worker will have access to `cores` number of threads.

    Returns:
        str or None: A filename if the requested resources could be matched to an
            existing scheduler, or None otherwise.

    """
    if match == "above":
        comp = ge
    elif match == "same":
        comp = eq
    else:
        raise ValueError(f"Invalid argument for 'match' given: {repr(match)}.")

    specs["cores"] = specs.get("cores", DEFAULTS["cores"])
    specs["memory"] = pbs_format_bytes_ceil(
        parse_bytes(specs.get("memory", DEFAULTS["memory"]))
    )
    specs["walltime"] = specs.get("walltime", DEFAULTS["walltime"])

    for scheduler_file in os.listdir(SCHEDULER_DIR):
        with open(os.path.join(SCHEDULER_DIR, scheduler_file)) as f:
            worker_specs = json.load(f)["worker_specs"]
        for spec, requested_value in specs.items():
            try:
                stored_value = worker_specs[spec]
            except KeyError:
                logging.warning(
                    f"The resource specification at {scheduler_file} did not specify a "
                    f"value for '{spec}'."
                )
                break

            if spec == "memory":
                requested_value = parse_bytes(requested_value)
                stored_value = parse_bytes(stored_value)

            if not comp(stored_value, requested_value):
                # Try the next scheduler file.
                break
        else:
            # If all comparisons succeeded, return the scheduler file.
            return os.path.join(SCHEDULER_DIR, scheduler_file)


class CX1PBSJob(PBSJob):
    @wraps(PBSJob.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Modify the '_command_template' in order to specify a local scheduler port,
        # will usually be different from the original scheduler port. Within the job
        # script, this will be forwarded to the proper scheduler port using local port
        # forwarding.
        command_template = self._command_template.split()

        shutdown_seconds = kwargs.get("shutdown_seconds", 120)
        timeout = walltime_seconds(kwargs["walltime"]) - shutdown_seconds

        command_template[3] = "localhost:$LOCALSCHEDULERPORT"
        self._command_template = " ".join(
            [
                # Allow 2 minutes for setting up and cleaning up.
                "timeout",
                "--preserve-status",
                f"{timeout}s",
            ]
            + command_template
        )
        # Kill the local worker port sync process.
        self._command_template += "\nkill $SYNCWORKERPID"
        # Allow for the cleanup actions initiated by the above to finish.
        self._command_template += f"\nsleep {shutdown_seconds}"


class CX1Cluster(PBSCluster):
    @wraps(PBSCluster.__init__)
    def __init__(self, *args, verbose_ssh=False, **kwargs):
        # This will be overwritten later, and will be used to get a constant filename.
        self._scheduler_file = None

        # This where the scheduler is run, and thus where ports have to be forwarded to.
        hostname = socket.gethostname()

        # First place any args into kwargs, then update relevant entries in kwargs to
        # enable operation on CX1.
        bound_args = signature(super().__init__).bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        mod_kwargs = bound_args.arguments
        # Use kwargs as well (otherwise the whole kwargs dictionary would be used
        # as another 'kwargs' keyword argument instead of the arguments therein).
        mod_kwargs.update(mod_kwargs.pop("kwargs"))

        if mod_kwargs.get("processes", 1) > 1:
            raise ValueError(
                "Only one worker process per job is supported (since worker ports have "
                "to be known to be forwarded)."
            )
        mod_kwargs["processes"] = 1
        nanny = mod_kwargs["nanny"] = mod_kwargs.get("nanny", True)

        # Set default parameters.
        mod_kwargs["cores"] = mod_kwargs.get("cores", DEFAULTS["cores"])
        mod_kwargs["memory"] = pbs_format_bytes_ceil(
            parse_bytes(mod_kwargs.get("memory", DEFAULTS["memory"]))
        )
        mod_kwargs["walltime"] = mod_kwargs.get("walltime", DEFAULTS["walltime"])

        # Get the number of workers.
        n_workers = mod_kwargs["n_workers"]
        if not n_workers > 0:
            logger.warning(f"Expected a positive number of workers, got {n_workers}.")

        scheduler_port = get_ports()[0]
        ssh_opts = " ".join(
            (
                "-NT",
                "-o StrictHostKeyChecking=no",
                "-o BatchMode=yes",
                "-o ServerAliveInterval=120",
                "-o ServerAliveCountMax=6",
            )
            + (("-vvv",) if verbose_ssh else ())
        )

        valid_ports_exec = (
            "/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/valid-ports"
        )

        file_id = "".join(choice(ascii_lowercase) for i in range(20))
        self.sync_stdout_file = open(f"sync_stdout_{file_id}.log", "w")
        self.sync_stderr_file = open(f"sync_stderr_{file_id}.log", "w")
        self.port_file = os.path.join(os.getcwd(), f"port_sync_{file_id}.db")

        initial_timeout = mod_kwargs.get("initial_timeout", 120)
        poll_interval = mod_kwargs.get("poll_interval", 10)

        sync_worker_ports_exec = (
            "/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/"
            f"sync-worker-ports --data-file {self.port_file} "
            f"--initial-timeout {initial_timeout} --poll-interval {poll_interval} "
            f"--ssh-opts='{ssh_opts}'" + (" --nanny" if nanny else "")
        )

        get_nanny_port_script = dedent(
            f"""
            # Get nanny port.
            #
            NANNYPORT=$({sys.executable}<<EOF
            with open("$WORKERPORTFILE") as f:
                print(f.read().strip().split()[1])
            EOF
            )
            #
            echo $(date): Got nanny port $NANNYPORT.
            #
            """
            if nanny
            else ""
        )

        mod_kwargs.update(
            job_cls=CX1PBSJob,
            extra=list(mod_kwargs.get("extra", []))
            + "--worker-port $WORKERPORT --no-dashboard".split()
            + ("--nanny-port $NANNYPORT".split() if nanny else []),
            # NOTE: Simple ssh, NOT autossh is used below, since using autossh
            # resulted in the connection being dropped repeatedly as it was
            # overzealously restarted.
            env_extra=f"""
JOBID="${{PBS_JOBID%%.*}}"
echo $(date): JOBID $JOBID on host $(hostname).
echo $(date): Getting ports.
read LOCALSCHEDULERPORT <<< $({valid_ports_exec} 1)
#
echo $(date): "Forwarding local scheduler port $LOCALSCHEDULERPORT to {hostname}."
ssh {ssh_opts} -L localhost:$LOCALSCHEDULERPORT:localhost:{scheduler_port} {hostname} &
#
# For the worker ports, extra care needs to be taken since these need to be the same
# on the worker and scheduler node. So we need to check that the port is unused on
# both! The same goes for the other worker ports, since those need to be forwarded as
# well to enable inter-worker communication.
#
# The sync-worker-ports executable is responsible for SSH forwarding of worker
# ports. Its PID is stored for later cleanup.
#
WORKERPORTFILE=${{JOBID}}_worker_port
echo $(date): Starting worker port sync in background.
echo $(date): Expecting worker port at file $WORKERPORTFILE.
{sync_worker_ports_exec} --output-file $WORKERPORTFILE &
SYNCWORKERPID=$!
#
# Wait for the worker sync program to write the worker port to the file.
#
echo $(date): Waiting for worker port.
WORKERPORT=$({sys.executable}<<EOF
import os
from time import sleep
while not os.path.isfile("$WORKERPORTFILE"):
    sleep(1)
with open("$WORKERPORTFILE") as f:
    ports = f.read().strip().split()
    if {nanny}:
        if len(ports) != 2:
            # We expected 2 ports (worker & nanny).
            print(-1)
    else:
        if len(ports) != 1:
            # We expected 1 port (worker only).
            print(-1)
    print(ports[0])
EOF
)
#
echo $(date): Got worker port $WORKERPORT.
#
# If the port is -1, this indicates an error.
#
if [[ $WORKERPORT == "-1" ]]; then
    echo Exiting, as an error occurred during port forwarding.
    exit 1
fi
#
{get_nanny_port_script}
echo $(date): Removing worker port file $WORKERPORTFILE.
rm "$WORKERPORTFILE"
#
sleep 1
echo $(date): Local processes:
pgrep -afu ahk114
#
echo $(date): Finished running ssh, starting dask-worker now.
""".strip().split(
                "\n"
            )
            + list(mod_kwargs.get("env_extra", ())),
            scheduler_options=dict(
                mod_kwargs.get("scheduler_options", ())
                if mod_kwargs.get("scheduler_options", ()) is not None
                else (),
                port=scheduler_port,
            ),
        ),
        super().__init__(**mod_kwargs)

        os.makedirs(SCHEDULER_DIR, exist_ok=True)
        info = self.scheduler_info
        info["worker_specs"] = mod_kwargs.copy()
        # When saving the worker specs, we carry about little other than the type of
        # the worker class.
        info["worker_specs"]["job_cls"] = mod_kwargs["job_cls"].__name__

        with open(self.scheduler_file, "w") as f:
            json.dump(info, f, indent=2)

        # In order for the above worker port forwarding to work, we need to start a
        # scheduler process which will check if the ports suggested by the workers are
        # available on the host running the scheduler. This will keep running until
        # the cluster shuts down.
        logger.info("Starting worker port synchronisation.")
        logger.info(
            f"Logging output to {self.sync_stdout_file.name} and "
            f"{self.sync_stderr_file.name}."
        )
        self.worker_sync_proc = Popen(
            shlex.split(
                f"{sync_worker_ports_exec} --scheduler --output-file scheduler_output"
            ),
            stdout=self.sync_stdout_file,
            stderr=self.sync_stderr_file,
        )

    def __cleanup(self):
        os.remove(self.scheduler_file)
        self.worker_sync_proc.kill()
        self.sync_stdout_file.close()
        self.sync_stderr_file.close()

    @wraps(PBSCluster.close)
    def close(self, *args, **kwargs):
        self.__cleanup()
        super().close(*args, **kwargs)

    @property
    def scheduler_file(self):
        """Get a unique filename for the current cluster scheduler."""
        # If this is the first time this is being run.
        if self._scheduler_file is None:
            fname = self.scheduler_info["id"] + ".json"
            assert fname not in os.listdir(
                SCHEDULER_DIR
            ), f"Scheduler file {fname} is already present."
            self._scheduler_file = os.path.join(SCHEDULER_DIR, fname)
        return self._scheduler_file

    @wraps(PBSCluster.scale)
    def scale(self, *args, **kwargs):
        logger.warning(cluster_size_error_msg)
        super().scale(*args, **kwargs)

    @wraps(PBSCluster.adapt)
    def adapt(self, *args, **kwargs):
        logger.warning(cluster_size_error_msg)
        super().adapt(*args, **kwargs)
