# -*- coding: utf-8 -*-
import atexit
import json
import logging
import math
import os
import re
import shlex
import socket
import sys
from contextlib import contextmanager
from copy import deepcopy
from functools import partial, wraps
from getpass import getuser
from inspect import signature
from operator import eq, ge, lt
from random import choice
from string import ascii_lowercase
from subprocess import Popen, check_call, check_output
from tempfile import NamedTemporaryFile
from textwrap import dedent
from time import sleep
from urllib.parse import urlparse, urlunparse

from dask.distributed import Client
from dask.utils import parse_bytes
from dask_jobqueue.pbs import PBSCluster, PBSJob, pbs_format_bytes_ceil
from joblib import parallel_backend

from ..ports import get_ports
from ..qstat import get_ncpus

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


class SchedulerConnectionError(DaskCX1Error):
    """Raised when a connection to a scheduler could not be established."""


def multiline(s, strip_all_indents=False):
    if strip_all_indents:
        return " ".join([dedent(sub) for sub in s.strip().split("\n")])
    else:
        return dedent(s).strip().replace("\n", " ")


strip_multiline = partial(multiline, strip_all_indents=True)


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


def get_client(fallback=False, fallback_cores=None, fallback_threaded=False, **specs):
    """Try to connect to an existing Dask scheduler with the supplied specs.

    Args:
        specs: Worker resource specifications, eg. cores=10, memory="8GB". The
            scheduler's cluster will have access to a certain number of such workers,
            where each worker will have access to `cores` number of threads.
        fallback (bool or int): If True, allow falling back to a Dask LocalCluster.
        fallback_cores (int): The number of cores to use for the fallback
            LocalCluster. If not given (None), `wildfires.qstat.get_ncpus()` will be
            used.
        fallback_threaded (bool): If True, only a single multi-threaded fallback
            LocalCluster worker will be created (number of cores as above).

    Returns:
        distributed.client.Client: Client used to connect to the scheduler.

    Raises:
        FoundSchedulerError: If no matching scheduler could be found and `fallback` is
            False.
        SchedulerConnectionError: If a matching scheduler was found but a connection
            could not be established, and `fallback` is False.

    """
    try:
        return get_remote_client(**specs)
    except DaskCX1Error:
        if not fallback:
            raise
        if fallback_cores is not None:
            cores = fallback_cores
        else:
            cores = get_ncpus()
        if fallback_threaded:
            return Client(n_workers=1, threads_per_worker=cores)
        else:
            return Client(n_workers=cores, threads_per_worker=1)


def get_remote_client(**specs):
    """Try to connect to an existing remote Dask scheduler with the supplied specs.

    Args:
        specs: Worker resource specifications, eg. cores=10, memory="8GB". The
            scheduler's cluster will have access to a certain number of such workers,
            where each worker will have access to `cores` number of threads.

    Returns:
        distributed.client.Client: Client used to connect to the scheduler.

    Raises:
        FoundSchedulerError: If no matching scheduler could be found.
        SchedulerConnectionError: If a matching scheduler was found, but a connection
            could not be established.

    """
    logger.info(f"Trying to find Dask scheduler with minimum specs {specs}.")
    scheduler_file = get_scheduler_file(**specs)

    if scheduler_file is None:
        raise FoundSchedulerError(f"No scheduler with minimum specs {specs} found.")

    with open(scheduler_file) as f:
        scheduler_info = json.load(f)

    cluster_address = urlparse(scheduler_info["address"])
    cluster_netloc = cluster_address.netloc
    cluster_host, _, cluster_port = cluster_netloc.partition(":")

    cluster_hostname = scheduler_info["worker_specs"]["cluster_hostname"]
    our_hostname = socket.getfqdn()

    try:
        logger.info(f"Found scheduler at {scheduler_file}.")
        # Try connecting to the Client. If this does not work, we will try port
        # forwarding.

        if cluster_hostname != our_hostname:
            # Check if port forwarding has been carried out previously.
            try:
                return Client(
                    urlunparse(
                        cluster_address._replace(
                            netloc=":".join(("localhost", cluster_port))
                        )
                    ),
                    timeout=10,
                )
            except OSError:
                logger.debug("No previous port forwarding could be used.")

        # Try using the actual address.
        return Client(scheduler_file=scheduler_file, timeout=10)
    except OSError:
        logger.warning(
            f"Could not connect to the scheduler at {scheduler_file} "
            "(no port forwarding)."
        )

    if cluster_hostname != our_hostname:
        # We are on a different host, so port forwarding might have to be used.
        logger.info(f"Starting local port forwarding to {cluster_netloc}.")
        ssh_command = (
            f"ssh -NT -L localhost:{cluster_port}:{cluster_netloc} "
            f"{getuser()}@{cluster_hostname} "
            "-o StrictHostKeyChecking=no"
        )
        logger.debug(f"SSH command: {ssh_command}.")
        # TODO: Capture stderr.
        ssh = Popen(shlex.split(ssh_command))

        def kill_local_forward_ssh():
            logger.debug(f"Shutting down SSH local port forwarding ({ssh}).")
            ssh.kill()

        atexit.register(kill_local_forward_ssh)

        # Replace the host address with localhost, since we are now forwarding the
        # port.
        cluster_address = cluster_address._replace(
            netloc=":".join(("localhost", cluster_port))
        )

        try:
            # Try connecting to the Client.
            new_client_address = urlunparse(cluster_address)
            logger.debug(f"Connecting to client at {new_client_address}.")
            return Client(new_client_address, timeout=10)
        except OSError:
            logger.warning(
                f"Could not connect to the scheduler at {scheduler_file} after port "
                "forwarding."
            )

    raise SchedulerConnectionError(
        f"Could not connect to scheduler at {scheduler_file}."
    )


@contextmanager
def get_parallel_backend(fallback="loky", **specs):
    """Try to connect to an existing Dask scheduler with the supplied specs.

    The fallback is a local loky backend with the number of CPUs determined by the
    currently running job (if any, eg. a JupyterLab job), as reported
    `wildfires.utils.get_ncpus()`.

    Args:
        specs: Worker resource specifications, eg. cores=10, memory="8GB". The
            scheduler's cluster will have access to a certain number of such workers,
            where each worker will have access to `cores` number of threads.
        fallback (str): If not False (eg. the empty string ""), allow falling back to
            the backend specified by the supplied string, eg. "loky" or "sequential".
            Additionally, if `fallback` is "none", no joblib backend will be invoked
            at all.

    Yields:
        joblib.parallel.parallel_backend or None: Context manager to manage nested
            calls to `joblib.parallel.Parallel()`. None if `fallback='none'`.
        distributed Client or None: If a Dask scheduler was found, the associated
            distributed Client will be yielded. Otherwise this will be None.

    Raises:
        FoundSchedulerError: If no matching scheduler could be found and `not fallback`.

    Examples:
        >>> from joblib import Parallel, delayed
        >>> with get_parallel_backend(cores=999999) as (backend, client):
        ...     assert client is None, "Can't possible have that many cores!"
        ...     out = Parallel()(delayed(lambda x: x + 1)(i) for i in  range(4))
        >>> print(out)
        [1, 2, 3, 4]

    """
    try:
        # Make sure that the Client disconnects.
        with get_client(**specs) as client, parallel_backend(
            "dask", wait_for_workers_timeout=600
        ) as backend:
            yield backend, client
            # XXX: Prevent closed connection errors due to abrupt closing of the client.
            # TODO: Re-use of the same client.
            sleep(1)
    except (FoundSchedulerError, SchedulerConnectionError) as err:
        logger.warning(f"Could not connect to scheduler: {err}.")
        # If no scheduler could be found, or a connection could not be established.
        if not fallback:
            raise FoundSchedulerError(f"No scheduler with minimum specs {specs} found.")
        if fallback == "none":
            logger.info("Not using any backend.")
            yield None, None
        else:
            ncpus = get_ncpus()
            logger.info(f"Using {fallback} backend with {ncpus} jobs.")
            yield parallel_backend(fallback, n_jobs=ncpus), None


def get_scheduler_file(match="above", **specs):
    """Get a scheduler file describing an existing scheduler.

    The newest file matching the given `specs` will be returned.

    This file can then be used to connect to the scheduler like so:

    >>> from dask.distributed import Client  # doctest: +SKIP
    >>> scheduler_file = get_scheduler_file(cores=3, memory="10GB")  # doctest: +SKIP
    >>> client = Client(scheduler_file=scheduler_file)  # doctest: +SKIP

    The client can be used to submit tasks to the scheduler.

    A more robust way to get a Client is to use `get_client()`, which additionally
    attempts port forwarding to the scheduler's host if the connection does not
    succeed at first.

    Args:
        match {"above", "same", or "below"}: All resource specifications provided as
            keyword arguments (eg. 'cores=10') will be matched against the worker
            specifications in existing scheduler files. For example, if "above", all
            specifications must match or exceed the requested values for a match to be
            reported.
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
    elif match == "below":
        comp = lt
    else:
        raise ValueError(f"Invalid argument for 'match' given: {repr(match)}.")

    if not os.path.isdir(SCHEDULER_DIR):
        logger.info("Scheduler directory not found.")
        return

    specs["cores"] = specs.get("cores", DEFAULTS["cores"])
    specs["memory"] = pbs_format_bytes_ceil(
        parse_bytes(specs.get("memory", DEFAULTS["memory"]))
    )
    specs["walltime"] = specs.get("walltime", DEFAULTS["walltime"])

    for scheduler_file in sorted(
        os.listdir(SCHEDULER_DIR),
        key=lambda scheduler_file: os.path.getctime(
            os.path.join(SCHEDULER_DIR, scheduler_file)
        ),
        reverse=True,
    ):
        with open(os.path.join(SCHEDULER_DIR, scheduler_file)) as f:
            worker_specs = json.load(f)["worker_specs"]
        for spec, requested_value in specs.items():
            try:
                stored_value = worker_specs[spec]
            except KeyError:
                logger.warning(
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
    logger.info(f"Could not find a Dask scheduler ({match}) for specs {specs}.")


class CX1PBSJob(PBSJob):
    """Job that runs on CX1, assuming no direct inter-job communication is possible."""

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
        # TODO: Add code to terminate the dask-worker if an error code has been
        # written to $WORKERPORTFILE.
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
    """Cluster on CX1, assuming no direct inter-job communication is possible."""

    @wraps(PBSCluster.__init__)
    def __init__(self, *args, verbose_ssh=False, **kwargs):
        # This will be overwritten later, and will be used to get a constant filename.
        self._scheduler_file = None

        # This where the scheduler is run, and thus where ports have to be forwarded to.
        hostname = socket.getfqdn()

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

        # Directory where workers will place their counter files.
        self.counter_dir = os.path.join(os.getcwd(), f"counters_{file_id}")

        initial_timeout = mod_kwargs.get("initial_timeout", 120)

        # Set the port sync program up so that workers can be unresponsive for up to X
        # minutes before they are killed using 'qdel'. X is the product of the poll
        # interval (p_i) and the keepalive intervals (n_k), ie. X = p_i x k_i.

        # Since an unresponsive worker might block the database while it is
        # unresponsive, we need another way to signal to the scheduler that responsive
        # workers are still alive - this is done using files in a dedicated folder
        # wherein one file is used by each worker to store and increment a personal
        # counter which is then read by the scheduler.

        poll_interval = mod_kwargs.get("poll_interval", 30)
        keepalive_intervals = mod_kwargs.get("keepalive_intervals", 4)

        # Amount of time before a worker gives up waiting for an unresponsive worker.
        sqlite_timeout = mod_kwargs.get("sqlite_timeout", 10)

        sync_worker_ports_bin = (
            "/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/"
            "sync-worker-ports"
        )

        sync_worker_ports_exec = (
            strip_multiline(
                f"""{sync_worker_ports_bin}
                --data-file {self.port_file}
                --initial-timeout {initial_timeout}
                --poll-interval {poll_interval}
                --keepalive-intervals {keepalive_intervals}
                --sqlite-timeout {sqlite_timeout}
                --enable-qdel
                --counter-dir {self.counter_dir}
                --ssh-opts='{ssh_opts}'"""
            )
            + (" --nanny" if nanny else "")
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

        threads = math.floor(mod_kwargs["cores"] / mod_kwargs["processes"])

        mod_kwargs.update(
            job_cls=CX1PBSJob,
            extra=list(mod_kwargs.get("extra", []))
            + "--worker-port $WORKERPORT --no-dashboard".split()
            + f"--resources threads={threads}".split()
            + ("--nanny-port $NANNYPORT".split() if nanny else []),
            # NOTE: Simple ssh, NOT autossh is used below, since using autossh
            # resulted in the connection being dropped repeatedly as it was
            # overzealously restarted.
            env_extra=f"""
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
#
JOBID="${{PBS_JOBID%%.*}}"
echo $(date): JOBID $JOBID on host $(hostname).
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
    sleep(5)
sleep(5)
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
echo $(date): Getting local scheduler port.
read LOCALSCHEDULERPORT <<< $({valid_ports_exec} 1)
#
echo $(date): "Forwarding local scheduler port $LOCALSCHEDULERPORT to {hostname}."
ssh {ssh_opts} -L localhost:$LOCALSCHEDULERPORT:localhost:{scheduler_port} {hostname} &
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

        info["worker_specs"]["cluster_hostname"] = hostname

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
        print(f"Dashboard at: {self.dashboard_link}")

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


class CX1GeneralCluster(PBSCluster):
    """Cluster on CX1 in the general queue with direct inter-job communication.

    Start the cluster in its own job to let it request additional workers and so that
    the initial scheduler address can be used.

    """

    @wraps(PBSCluster.__init__)
    def __init__(self, *args, **kwargs):
        # This will be overwritten later, and will be used to get a constant filename.
        self._scheduler_file = None

        # This where the scheduler is run, and thus where ports have to be forwarded to.
        hostname = socket.getfqdn()

        # First place any args into kwargs, then update relevant entries in kwargs to
        # enable operation on CX1.
        bound_args = signature(super().__init__).bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        mod_kwargs = bound_args.arguments

        # Use kwargs as well (otherwise the whole kwargs dictionary would be used
        # as another 'kwargs' keyword argument instead of the arguments therein).
        mod_kwargs.update(mod_kwargs.pop("job_kwargs"))

        # Set default parameters.
        if mod_kwargs.get("cores") is not None and mod_kwargs.get("cores") != 32:
            raise ValueError("32 cores are needed in the general queue.")
        mod_kwargs["cores"] = 32

        if mod_kwargs.get("memory") is None:
            mod_kwargs["memory"] = "62GiB"
        mod_kwargs["memory"] = pbs_format_bytes_ceil(parse_bytes(mod_kwargs["memory"]))

        if (
            mod_kwargs.get("interface") is not None
            and mod_kwargs.get("interface") != "eth0"
        ):
            raise ValueError(
                "The 'eth0' interface should be used for 'general' class jobs."
            )
        mod_kwargs["interface"] = "eth0"

        if mod_kwargs.get("processes") is None:
            mod_kwargs["processes"] = 1

        # Get the number of workers.
        n_workers = mod_kwargs["n_workers"]
        if not n_workers > 0:
            logger.warning(f"Expected a positive number of workers, got {n_workers}.")

        if mod_kwargs.get("walltime") is None:
            mod_kwargs["walltime"] = DEFAULTS["walltime"]

        lifetime = mod_kwargs.get(
            "lifetime",
            walltime_seconds(mod_kwargs["walltime"])
            - mod_kwargs.get("shutdown_seconds", 120),
        )

        stagger = mod_kwargs.get(
            "lifetime_stagger",
            max(60, min(600, round(0.1 * walltime_seconds(mod_kwargs["walltime"])))),
        )

        threads = mod_kwargs.get(
            "threads_per_worker",
            math.floor(mod_kwargs["cores"] / mod_kwargs["processes"]),
        )
        if threads < 1:
            raise ValueError(
                f"'Threads' was below 1 ({threads}, from "
                f"{mod_kwargs['cores'], mod_kwargs['processes']})"
            )
        logger.debug(f"{threads} thread(s) per worker.")

        mod_kwargs.update(
            job_cls=PBSJob,
            extra=list(mod_kwargs.get("extra", []))
            + "--no-dashboard".split()
            + f"--resources threads={threads}".split()
            + f"--lifetime {lifetime}".split()
            + f"--lifetime-stagger {stagger}".split(),
            env_extra=list(mod_kwargs.get("env_extra", ()))
            + f"""
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
#
JOBID="${{PBS_JOBID%%.*}}"
echo $(date): JOBID $JOBID on host $(hostname).
echo $(date): Local processes:
pgrep -afu ahk114
#
""".strip().split(
                "\n"
            ),
            scheduler_options=dict(
                mod_kwargs.get("scheduler_options", ())
                if mod_kwargs.get("scheduler_options", ()) is not None
                else ()
            ),
        ),
        super().__init__(**mod_kwargs)

        os.makedirs(SCHEDULER_DIR, exist_ok=True)
        info = self.scheduler_info
        info["worker_specs"] = mod_kwargs.copy()
        # When saving the worker specs, we carry about little other than the type of
        # the worker class.
        info["worker_specs"]["job_cls"] = mod_kwargs["job_cls"].__name__

        info["worker_specs"]["cluster_hostname"] = hostname

        with open(self.scheduler_file, "w") as f:
            json.dump(info, f, indent=2)

        print(f"Dashboard at: {self.dashboard_link}")

    def __cleanup(self):
        os.remove(self.scheduler_file)

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


class CX1GeneralArrayCluster(CX1GeneralCluster):
    """Cluster on CX1 in the general queue with direct inter-job communication.

    Start the cluster in its own job to let it request additional workers and so that
    the initial scheduler address can be used.

    Jobs are submitted using array jobs.

    """

    @wraps(CX1GeneralCluster.__init__)
    def __init__(self, *args, **kwargs):
        self.array_jobs = {}
        super().__init__(*args, **kwargs)

    @wraps(PBSCluster.scale)
    def scale(self, n_workers):
        if n_workers == 0:
            # Terminate all jobs.
            concat_jobs = " ".join(self.array_jobs)
            logger.debug(f"Terminating jobs {concat_jobs}.")
            check_call(shlex.split(f"qdel {concat_jobs}"))
            return

        if n_workers == 1:
            raise ValueError("At least 2 jobs are required for array job submission.")

        kwargs = deepcopy(self._kwargs)
        # Add the array job directive (indices are inclusive).
        kwargs["env_extra"] = [f"#PBS -J 0-{n_workers - 1}"] + list(
            kwargs.get("env_extra", ())
        )
        with NamedTemporaryFile(
            prefix=f"dask_general_array_job_", suffix=".sh"
        ) as job_file:
            with open(job_file.name, "w") as f:
                f.write(
                    self.job_cls(
                        scheduler=self.scheduler.address,
                        name="$PBS_ARRAY_INDEX",
                        **kwargs,
                    ).job_script()
                )

            job_str = (
                check_output(shlex.split(f"qsub -V {job_file.name}")).decode().strip()
            )
            logger.debug(f"Submitted job {job_str}.")

            # Record this job.
            self.array_jobs[job_str] = n_workers
