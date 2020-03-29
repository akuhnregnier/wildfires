# -*- coding: utf-8 -*-
import atexit
import json
import logging
import os
import sys
from functools import wraps
from inspect import signature
from operator import eq, ge

from dask.distributed import Client
from dask.utils import parse_bytes
from joblib import parallel_backend

from dask_jobqueue.pbs import PBSCluster, PBSJob, pbs_format_bytes_ceil

from .ports import get_ports
from .utils import get_ncpus

logger = logging.getLogger(__name__)

SCHEDULER_DIR = os.path.expanduser(os.path.join("~", "schedulers"))

DEFAULTS = {"cores": 1, "memory": "6GB", "walltime": "10:00:00"}


class Dask_CX1_Error(Exception):
    """Base class for exceptions in the dask_cx1 module."""


class FoundSchedulerError(Dask_CX1_Error):
    """Raised when no scheduler matching the requested specifications could be found."""


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

            if not comp(requested_value, stored_value):
                # Try the next scheduler file.
                break
        else:
            # If all comparisons succeeded, return the scheduler file.
            return scheduler_file


class CX1PBSJob(PBSJob):
    @wraps(PBSJob.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Modify the '_command_template' in order to specify a local scheduler port,
        # will usually be different from the original scheduler port. Within the job
        # script, this will be forwarded to the proper scheduler port using local port
        # forwarding.
        command_template = self._command_template.split()
        command_template[3] = "localhost:$LOCALSCHEDULERPORT"
        self._command_template = " ".join(command_template)


class CX1Cluster(PBSCluster):
    @wraps(PBSCluster.__init__)
    def __init__(self, *args, verbose_ssh=False, **kwargs):
        # This will be overwritten later, and will be used to get a constant filename.
        self._scheduler_file = None

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

        # Set default parameters.
        mod_kwargs["cores"] = mod_kwargs.get("cores", DEFAULTS["cores"])
        mod_kwargs["memory"] = pbs_format_bytes_ceil(
            parse_bytes(mod_kwargs.get("memory", DEFAULTS["memory"]))
        )
        mod_kwargs["walltime"] = mod_kwargs.get("walltime", DEFAULTS["walltime"])

        scheduler_port = get_ports()[0]
        ssh_extra = "-vvv" if verbose_ssh else ""
        mod_kwargs.update(
            job_cls=CX1PBSJob,
            extra=list(mod_kwargs.get("extra", []))
            + "--worker-port $WORKERPORT".split(),
            # NOTE: Simple ssh, NOT autossh is used below, since using autossh
            # resulted in the connection being dropped repeatedly as it was
            # overzealously restarted.
            env_extra=(
                f"PYTHON={sys.executable}",
                f"SCHEDULERADDRESS='localhost:{scheduler_port}'",
                "echo 'Getting ports'",
                "read LOCALSCHEDULERPORT <<< $(/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/valid-ports 1)",
                'echo "local scheduler port:$LOCALSCHEDULERPORT"',
                # For the worker port, extra care needs to be taken since this port
                # needs to be the same on the worker and scheduler node. So we need to
                # check that the port is unused on both!
                "WORKERPORT=$($PYTHON << EOF",
                "from subprocess import call",
                "from wildfires.ports import get_ports",
                "port_found = False",
                "cycles = 0",
                # Check if the local port would also work on the remote, ie. scheduler
                # node. Check for return code 1 which rules out some ssh error codes.
                "while True:",
                "   port = get_ports()[0]",
                '   if call(["ssh", "-o BatchMode=yes", "login-7", f"nc -z localhost {port}"]) == 1:',
                "       print(port)",
                "       break",
                "   cycles += 1",
                "   if cycles > 20:",
                '       raise RuntimeError("No valid port could be found.")',
                "EOF",
                ")",
                "PORTCODE=$?",
                "if [ $PORTCODE -eq 0 ]",
                "then",
                'echo "worker port:$WORKERPORT"',
                "else",
                'echo "Could not get worker port." >&2',
                "exit $PORTCODE",
                "fi",
                "echo 'Running ssh'",
                f"ssh {ssh_extra} -N -T -L localhost:$LOCALSCHEDULERPORT:$SCHEDULERADDRESS -R localhost:$WORKERPORT:localhost:$WORKERPORT login-7 -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=120 -o ServerAliveCountMax=6 &",
                "echo 'Finished running ssh, starting dask-worker now'",
                "sleep 1",
                "echo 'Local processes:'",
                "pgrep -afu ahk114",
            )
            + tuple(mod_kwargs.get("env_extra", ())),
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

        atexit.register(self.__cleanup)

    def __cleanup(self):
        os.remove(self.scheduler_file)

    @wraps(PBSCluster.close)
    def close(self, *args, **kwargs):
        self.__cleanup()
        atexit.unregister(self.__cleanup)
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
