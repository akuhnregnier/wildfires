# -*- coding: utf-8 -*-
"""To be called from from several nodes to coordinate worker port numbers.

Identical arguments must be used for each set of workers to coordinate.

NOTE:

    This relies on a shared filesystem!

During the initial timeout (relative to the start of the first instance) worker
port numbers are negotiated between active workers such that all used ports are open
on all involved nodes.

After the timeout, each process prints its own worker port number to stdout and
initiates the needed port forwarding commands.

The local worker port number has to be forwarded to the scheduler via remote port
forwarding.

The other ports (if any) have to be forwarded to the scheduler via local port
forwarding.

The program will only forward ports once per host.

Note that the database file is not removed automatically upon finishing, and so
should be cleaned up by a coordinating process (most likely the Cluster init).

"""
import atexit
import logging
import math
import os
import re
import shlex
import signal
import socket
import sqlite3
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict, defaultdict, namedtuple
from datetime import datetime
from functools import reduce
from operator import add
from sched import scheduler
from sqlite3 import IntegrityError, OperationalError
from subprocess import Popen
from textwrap import dedent, wrap
from time import monotonic, time

import pandas as pd

from wildfires.logging_config import enable_logging

from .ports import get_ports

logger = logging.getLogger(__name__)


class PortSyncError(Exception):
    """Base class for exceptions raised while syncing worker ports."""


class RemoteForwardOutputError(PortSyncError):
    """Raised when a port has already been remote-forwarded by not previously output."""


class ChangedPortsError(PortSyncError):
    """Raised when ports changed during a check."""


class SchedulerError(PortSyncError):
    """Raised when a scheduling error is encountered."""


def multiline(s, strip_all_indents=False):
    if strip_all_indents:
        return " ".join([dedent(sub) for sub in s.strip().split("\n")])
    else:
        return dedent(s).strip().replace("\n", " ")


class Scheduler:
    def __init__(self, interval, action, args=None, kwargs=None, max_iter=math.inf):
        """Scheduler for repeated execution of an action.

        Args:
            interval (float): Interval between actions in seconds.
            action (callable): Action.
            args (tuple): Positional arguments.
            kwargs (dictionary): Keyword arguments.
            max_iter (int or float): Maximum number of iterations.

        """
        self.scheduler = scheduler()

        self.interval = interval
        self.action = action
        self.args = args if args is not None else ()
        self.kwargs = kwargs if kwargs is not None else {}
        self.max_iter = max_iter

        self.first_time = None
        self.iterations = 0
        self.n_intervals = 0
        self.next_event = None

    def run(self):
        """Start running the action."""
        self.scheduler.enter(0, 0, self._execute_action)
        self.scheduler.run()
        return self

    def _execute_action(self):
        """Execute the action and schedule the next execution."""
        if self.first_time is None:
            # Keep track of when the first execution was.
            self.first_time = monotonic()

        # Actually execute the action.
        self.action(*self.args, **self.kwargs)
        self.iterations += 1

        # Schedule the next action.

        # Number of intervals needed to get to the next timestep. This relies on the
        # fact that that the actual time taken to get here is longer than
        # `self.interval`. Note that a small offset is applied to account for events
        # which finished just before the start of a new interval.
        # If the time to the next interval is too short, the next scheduled time could
        # have passed by the time we call `enterabs()`. The required offset needed to
        # circumvent this is machine and load dependent.
        next_intervals = math.ceil(
            ((monotonic() + 5e-4) - self.first_time) / self.interval
        )
        interval_diff = next_intervals - self.n_intervals

        if interval_diff != 1:
            if interval_diff < 1:
                logger.error(
                    f"Calculated {interval_diff} required intervals "
                    f"at iteration {self.iterations}."
                )
            # Warn, and fix number of intervals.
            fixed_interval_diff = max((1, interval_diff))
            logger.error(
                f"Need {fixed_interval_diff} intervals at iteration {self.iterations}."
                f"Consider increasing the interval (currently {self.interval:0.2e} s)."
            )
            interval_diff = fixed_interval_diff

        next_time = self.first_time + (self.n_intervals + interval_diff) * self.interval

        if next_time <= monotonic():
            raise SchedulerError(
                f"Next iteration scheduled for a time in the past. "
                f"Consider increasing the interval (currently {self.interval:0.2e} s)."
            )

        self.n_intervals = next_intervals

        if self.iterations < self.max_iter:
            self.next_event = self.scheduler.enterabs(
                next_time, 0, self._execute_action
            )

    def cancel(self):
        """Cease to continually execute `action`."""
        if self.next_event is not None:
            self.scheduler.cancel(self.next_event)
            self.next_event = None


class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, name, hostname):
        super(LoggerAdapter, self).__init__(logger, {})

        self.worker_name = name
        self.hostname = hostname

    def process(self, msg, kwargs):
        return f"\n[{self.worker_name}:{self.hostname}] {msg}", kwargs


_arg_tuple = namedtuple("Arg", ["type", "help", "default"])


def arg_tuple(type_string, help_string, default_value):
    """Get a descriptive argument namedtuple with concatenated help string."""
    # Apply `multiline` to the help strings.
    help_string = multiline(help_string, strip_all_indents=True)
    return _arg_tuple(type_string, help_string, default_value)


# Type, help and default values for PortSync arguments.
port_sync_args = OrderedDict(
    data_file=arg_tuple(
        "str",
        """Path to the file used to store each worker's desired worker port and other
        information required for this process.""",
        os.path.join(os.path.expanduser("~"), "worker_dir", "port_sync.db"),
    ),
    initial_timeout=arg_tuple(
        "float",
        """Timeout in seconds before the port synchronisation is concluded. After the
        timeout, any additional workers will be tied into the cluster if the existing
        ports are available on the new node and rejected otherwise. The shortest
        timeout wins.""",
        300,
    ),
    poll_interval=arg_tuple(
        "float", "Interval in seconds between file read operations.", 10
    ),
    scheduler=arg_tuple(
        "bool",
        """If True, this process should be on the scheduler node. In a set of
        processes running this program, only one should be run with `scheduler=True`.
        The scheduler will not suggest any ports and only report whether the ports
        suggested by the workers (run with `scheduler=False`) are usable on the
        scheduler node.""",
        False,
    ),
    debug=arg_tuple(
        "bool", "If True, only print out SSH commands instead of executing them.", False
    ),
    output_file=arg_tuple(
        "str",
        """Path to the file where the worker port will be written to. If not supplied,
        the worker port will be printed to stdout instead.""",
        None,
    ),
    ssh_opts=arg_tuple(
        "iterable of str", "A series of SSH command line options.", ("-NT",)
    ),
    hostname=arg_tuple("str", "If given, override automatic hostname detection.", None),
    port=arg_tuple("int", "If given, override automatic port determination.", None),
    keepalive_intervals=arg_tuple(
        "int",
        """The number of intervals for which a worker's counter may stay constant
        before it is cleared.""",
        2,
    ),
    nanny=arg_tuple(
        "bool",
        "Determine and remote-forward a nanny port in addition to the worker port.",
        False,
    ),
    nanny_port=arg_tuple(
        "int",
        "If given, override automatic nanny port determination. "
        "Requires `--nanny` or `nanny=True`.",
        None,
    ),
)

width = 88
tab = 4
args_help = ""
for (name, (arg_type, arg_help, _)) in port_sync_args.items():
    arg_help = f"{name} ({arg_type}): {arg_help}"
    parts = wrap(arg_help, width - tab)
    lines = [" " * tab + parts[0]]
    if len(parts) > 1:
        lines.append(
            "\n".join(
                [
                    " " * 2 * tab + wrapped
                    for wrapped in wrap(" ".join(parts[1:]), width - 2 * tab)
                ]
            )
        )
    args_help = args_help + "\n" + "\n".join(lines)

port_sync_doc = __doc__ + f"Args:{args_help}\n"


class PortSync:
    __doc__ = port_sync_doc

    def __init__(
        self,
        data_file=port_sync_args["data_file"].default,
        initial_timeout=port_sync_args["initial_timeout"].default,
        poll_interval=port_sync_args["poll_interval"].default,
        scheduler=port_sync_args["scheduler"].default,
        debug=port_sync_args["debug"].default,
        output_file=port_sync_args["output_file"].default,
        ssh_opts=port_sync_args["ssh_opts"].default,
        hostname=port_sync_args["hostname"].default,
        port=port_sync_args["port"].default,
        keepalive_intervals=port_sync_args["keepalive_intervals"].default,
        nanny=port_sync_args["nanny"].default,
        nanny_port=port_sync_args["nanny_port"].default,
    ):
        if nanny_port is not None and not nanny:
            raise ValueError(f"`nanny_port={nanny_port}` given, but `nanny=False`.")

        self.start_time = time()

        self.initial_timeout = initial_timeout
        self.poll_interval = poll_interval
        self.is_scheduler = scheduler
        self.debug = debug
        if not self.debug:
            if output_file is None:
                raise ValueError("An output file needs to be supplied.")
            else:
                output_file = os.path.expanduser(output_file)
        self.output_file = output_file
        if not all(isinstance(opt, str) for opt in ssh_opts):
            raise ValueError(f"All SSH options must be strings. Got {repr(ssh_opts)}.")
        self.ssh_opts = ssh_opts
        self.nanny = nanny

        self.ssh_procs = []
        self.output_already = False
        self._locked_port = False

        # This will be the unique (worker) identifier.
        self.name = (
            "scheduler" if self.is_scheduler else os.environ["PBS_JOBID"].split(".")[0]
        )

        # The hostname is important as ports only need to be forwarded once per host,
        # since intra-host communication should be possible without port forwarding.
        self.hostname = hostname if hostname is not None else socket.gethostname()

        # TODO: Need to configure this logger to make this work.
        # logger = logging.getLogger(self.__class__.__name__)
        self.logger = LoggerAdapter(logger, self.name, self.hostname)

        self.loop_scheduler = Scheduler(self.poll_interval, self._sync_iteration)

        self.logger.info("Starting port number sync.")

        if not os.path.isfile(data_file):
            self.logger.info(f"Database file {data_file} was not found.")
        self.con = sqlite3.connect(data_file, timeout=40)

        # Ensure a graceful exit when possible. Our own entry will be cleared from the
        # database, SSH processes killed, and the database connection closed.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: self.abort(code=signum))
        atexit.register(self.close)

        # Create the necessary tables.
        # Using foreign key constraints, ensure that local and remote forwarding
        # entries have corresponding pre-existing worker name and port entries in the
        # 'workers' table.
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS timeout_exceeded (
                exceeded BOOLEAN UNIQUE
            );
            CREATE TABLE IF NOT EXISTS workers (
                name VARCHAR NOT NULL UNIQUE,
                hostname VARCHAR NOT NULL,
                port INTEGER NOT NULL UNIQUE,
                ready BOOLEAN,
                invalid_port BOOLEAN,
                counter INTEGER NOT NULL,
                nanny_port INTEGER UNIQUE
            );
            CREATE TABLE IF NOT EXISTS local_forwards (
                name VARCHAR NOT NULL,
                port INTEGER NOT NULL,
                FOREIGN KEY(name) REFERENCES workers (name),
                FOREIGN KEY(port) REFERENCES workers (port)
            );
            CREATE TABLE IF NOT EXISTS remote_forwards (
                -- Each worker only remote-forwards their own port (and nanny port).
                name VARCHAR NOT NULL UNIQUE,
                port INTEGER NOT NULL UNIQUE,
                nanny_port INTEGER UNIQUE,
                FOREIGN KEY(name) REFERENCES workers (name),
                FOREIGN KEY(port) REFERENCES workers (port)
                FOREIGN KEY(nanny_port) REFERENCES workers (nanny_port)
            );

            -- Triggers raise an sqlite3.IntegrityError to abort a transaction.

            -- Ensure that there are no duplicates across BOTH the port and nanny_port
            -- columns. The triggers are duplicated for INSERT and UPDATE.

            CREATE TRIGGER IF NOT EXISTS ensure_unique_ports_insert
            BEFORE INSERT ON workers
            FOR EACH ROW
            BEGIN
                SELECT RAISE (ABORT, "Port already exists (worker port or nanny port).")
                WHERE NEW.port IN (
                    SELECT port
                    FROM workers
                    UNION
                    SELECT nanny_port
                    FROM workers
                ) OR NEW.nanny_port IN (
                    SELECT port
                    FROM workers
                    UNION
                    SELECT nanny_port
                    FROM workers
                );
            END;

            CREATE TRIGGER IF NOT EXISTS ensure_unique_ports_port_update
            BEFORE UPDATE OF port ON workers
            FOR EACH ROW
            BEGIN
                SELECT RAISE (ABORT, "Port already exists (worker port or nanny port).")
                WHERE NEW.port IN (
                    SELECT port
                    FROM workers
                    UNION
                    SELECT nanny_port
                    FROM workers
                );
            END;

            CREATE TRIGGER IF NOT EXISTS ensure_unique_ports_nanny_port_update
            BEFORE UPDATE OF nanny_port ON workers
            FOR EACH ROW
            BEGIN
                SELECT RAISE (ABORT, "Port already exists (worker port or nanny port).")
                WHERE NEW.nanny_port IN (
                    SELECT port
                    FROM workers
                    UNION
                    SELECT nanny_port
                    FROM workers
                );
            END;

            -- Prevent duplicated local-forwards on the same host (race condition).

            CREATE TRIGGER IF NOT EXISTS validate_existing_local_forwards
            BEFORE INSERT ON local_forwards
            FOR EACH ROW
            BEGIN
                SELECT RAISE (ABORT, "Duplicate local port forwarding attempted.")
                WHERE EXISTS (
                    SELECT hostname
                    FROM workers
                    WHERE name = NEW.name AND hostname IN (
                        SELECT hostname
                        FROM workers
                        INNER JOIN local_forwards USING (name)
                        WHERE local_forwards.port = NEW.port
                    )
                );
            END;

            -- Ensure each worker only remote-forwards their own port(s).

            CREATE TRIGGER IF NOT EXISTS only_own_remote_forward
            BEFORE INSERT ON remote_forwards
            FOR EACH ROW
            BEGIN
                SELECT RAISE (ABORT, "Attempted remote forward of foreign port.")
                WHERE EXISTS (
                    SELECT 1
                    FROM workers
                    WHERE port = NEW.port AND name != NEW.name
                ) OR (
                    SELECT 1
                    FROM workers
                    WHERE nanny_port = NEW.nanny_port AND name != NEW.name
                );
            END;

            -- Prevent invalidated ports from being forwarded.

            CREATE TRIGGER IF NOT EXISTS no_invalid_port_local_forward
            BEFORE INSERT ON local_forwards
            FOR EACH ROW
            BEGIN
                SELECT RAISE (ABORT, "Attempted local forward of invalid port.")
                WHERE EXISTS (
                    SELECT 1
                    FROM workers
                    WHERE port = NEW.port AND invalid_port = 1
                );
            END;

            CREATE TRIGGER IF NOT EXISTS no_invalid_port_remote_forward
            BEFORE INSERT ON remote_forwards
            FOR EACH ROW
            BEGIN
                SELECT RAISE (ABORT, "Attempted remote forward of invalid port.")
                WHERE EXISTS (
                    SELECT 1
                    FROM workers
                    WHERE port = NEW.port AND invalid_port = 1
                );
            END;

            -- Trigger to prevent already forwarded ports from being invalidated.

            CREATE TRIGGER IF NOT EXISTS no_forwarded_port_invalidation
            BEFORE UPDATE OF invalid_port ON workers
            FOR EACH ROW
            WHEN NEW.invalid_port = 1
            BEGIN
                SELECT RAISE (ABORT, "Attempted invalidation of forwarded port.")
                WHERE NEW.port IN (
                    SELECT port
                    FROM local_forwards
                    UNION
                    SELECT port
                    FROM remote_forwards
                    UNION
                    SELECT nanny_port
                    FROM remote_forwards
                );
            END;

            -- Trigger to prevent already invalidated ports from being added.

            CREATE TRIGGER IF NOT EXISTS no_invalid_port_addition
            BEFORE INSERT ON workers
            FOR EACH ROW
            WHEN NEW.invalid_port = 1
            BEGIN
                SELECT RAISE (ABORT, "Attempted addition of invalid port.")
                WHERE 1;
            END;

            -- Enable foreign key constraints.
            PRAGMA foreign_keys = ON;"""
        )

        def squeeze_row(cursor, row):
            """If there is only one column, only return that column's value."""
            if len(row) == 1:
                return row[0]
            return row

        self.con.row_factory = squeeze_row

        # Determine our port.
        if self.is_scheduler:
            if port is not None:
                self.logger.warning(
                    f"Worker `port` {port} supplied despite running as scheduler."
                )
            if nanny_port is not None:
                self.logger.warning(
                    f"Worker `nanny_port` {nanny_port} supplied despite running as "
                    "scheduler."
                )
            initial_port = -2
            initial_nanny_port = None

            # The scheduler is responsible for clearing unresponsive workers, and so
            # needs to keep track of all workers' counters.

            def get_zero():
                return 0

            self.counters = {}
            self.missed_counts = defaultdict(get_zero)
            self.keepalive_intervals = keepalive_intervals
        else:
            if port is None:
                self.logger.debug(
                    "No custom port supplied. Choosing one automatically."
                )
            else:
                self.logger.debug(f"Custom port {port} supplied.")
            initial_port = port

            if nanny:
                if nanny_port is None:
                    self.logger.debug(
                        "No custom nanny port supplied. Choosing one automatically."
                    )
                else:
                    self.logger.debug(f"Custom nanny port {nanny_port} supplied.")
            # If `nanny == False`, `nanny_port is None` is guaranteed.
            initial_nanny_port = nanny_port

        self.logger.debug("Inserting initial entry.")

        for _ in range(100):
            # Try for 100 times at most to set a new, unique port.
            try:
                with self.con:
                    self.con.execute(
                        "INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            self.name,
                            self.hostname,
                            get_ports()[0] if initial_port is None else initial_port,
                            False,
                            False,
                            0,
                            get_ports()[0]
                            if not self.is_scheduler
                            and self.nanny
                            and initial_nanny_port is None
                            else initial_nanny_port,
                        ),
                    )
                    if not self.is_scheduler:
                        # Notify other workers that they need to re-check ports.
                        self.notify()
                break  # Stop trying to get a new port.
            except IntegrityError as integrity_error:
                if re.match(
                    (
                        "(^Port already exists \(worker port or nanny port\).$"
                        "|^UNIQUE constraint failed: workers\..*$)"
                    ),
                    str(integrity_error),
                ):
                    self.logger.debug("Encountered port number clash.")
                else:
                    # Do not intercept other errors.
                    raise
        else:
            # No break encountered - the new port was always already present.
            self.abort("Could not find a suitable unique port. Exit (1).")

        self.logger.info(f"Proposing port {self.port}.")
        if self.nanny:
            self.logger.info(f"Proposing nanny port {self.nanny_port}.")

    @property
    def port(self):
        return self.con.execute(
            "SELECT port FROM workers WHERE name = ?", (self.name,)
        ).fetchone()

    @property
    def nanny_port(self):
        return self.con.execute(
            "SELECT nanny_port FROM workers WHERE name = ?", (self.name,)
        ).fetchone()

    def run(self):
        """Start synchronisation."""
        self.loop_scheduler.run()

    def _sync_iteration(self):
        """A synchronisation iteration which is looped using a scheduler."""
        try:
            if self.is_scheduler:
                # Check the worker counters to detect inactive workers.
                for worker_id, counter in self.con.execute(
                    "SELECT name, counter FROM workers WHERE name != 'scheduler'"
                ):
                    if worker_id in self.counters:
                        if counter == self.counters[worker_id]:
                            # The counter has not changed - a dead worker?
                            self.missed_counts[worker_id] += 1
                            if self.missed_counts[worker_id] > self.keepalive_intervals:
                                # The worker is unresponsive - assume it is dead.
                                print(f"Clearing worker: {worker_id}.", flush=True)
                                self.clear(worker_id)
                        else:
                            # Record the new counter value and reset the missed counts.
                            self.counters[worker_id] = counter
                            self.missed_counts[worker_id] = 0
                    else:
                        self.counters[worker_id] = counter
                # Print information about the current workers to stdout.
                print(datetime.now(), "(times may differ between hosts!)", flush=True)
                print(self, flush=True)
            if (
                all(self.con.execute("SELECT ready FROM workers"))
                and self.timeout_exceeded()
            ):
                # If every host has checked the ports and the initial sync timeout has
                # passed, we are ready to start issuing forwarding commands.
                self.logger.info(
                    f"All {self.n_workers} workers are ready. Our port: {self.port}."
                    + (f" Our nanny port: {self.nanny_port}." if self.nanny else "")
                )
                # Print ports.
                self.output()
            elif all(
                self.con.execute(
                    "SELECT ready FROM workers WHERE name = ?", (self.name,)
                )
            ):
                # Any worker on our host has already checked the ports locally.
                self.logger.info(
                    "Our host is ready. Waiting for other hosts or timeout."
                )
            else:
                # If we are not ready, check the ports on our host.
                self.logger.info("Checking ports.")
                self.check_ports()

            # Signal that we are still alive.
            self.increment_counter()
        except Exception:
            self.logger.exception("Continuing after encountering error.")

    def check_ports(self):
        if not self.locked_port and self.any_invalid_ports():
            # If we are still able to change our port, and if any port was unavailable.
            self.logger.warning("Encountered invalid port(s).")
            if self.port_is_invalid():
                self.logger.warning("Our own port is invalid.")
                self.get_new_port()
        else:
            # If all ports are available on other nodes, check that they work on ours.
            # ignoring the scheduler (since it doesn't have a real worker port).
            try:
                self._check_ports_locally()
            except OperationalError as operational_error:
                if str(operational_error) == "database is locked":
                    self.logger.warning("The database was locked for too long.")
                else:
                    # Only handle the locked database error.
                    raise

    def _check_ports_locally(self):
        """Check that ports work on our host.

        The scheduler is ignored since it doesn't have a real worker port.

        Raises:
            sqlite3.OperationalError: If another worker locked the database for too
                long while checking their own ports.

        """
        checked_ports = set()
        with self.con:
            # Lock the database while we check our ports to avoid other workers
            # checking the same ports concurrently on our host.
            # NOTE: This assumes all workers on the same node agree.
            self.con.execute("BEGIN EXCLUSIVE")
            for worker_id, port, nanny_port in self.con.execute(
                "SELECT name, port, nanny_port FROM workers WHERE name != 'scheduler'"
            ):
                checked_ports.update((port, nanny_port))
                self._port_available(worker_id, port)
                if nanny_port is not None:
                    self._port_available(worker_id, nanny_port)

            if not self.any_invalid_ports():
                try:
                    with self.con:
                        # Tell other workers that we have checked ports on our host.
                        self.con.execute(
                            "UPDATE workers SET ready = True WHERE hostname = ?",
                            (self.hostname,),
                        )
                        if (
                            set(
                                self.con.execute(
                                    """
                                    SELECT port
                                    FROM workers
                                    WHERE name != 'scheduler'
                                    UNION
                                    SELECT nanny_port
                                    FROM workers
                                    WHERE name != 'scheduler'"""
                                )
                            )
                            != checked_ports
                        ):
                            # XXX: With exclusive locking, this should never happen.
                            # We need to roll back the transaction, since we did not
                            # check all current ports  (ports may have been modified
                            # while we carried out our checks).
                            raise ChangedPortsError()
                except ChangedPortsError:
                    self.logger.warning("Ports changed during our availability check.")
                    self.notify(hostname=self.hostname)

    def _port_available(self, worker_id, port):
        """Check if a port is available on our host."""

        if port in self.host_forwarded_ports:
            # Skip ports that have already been forwarded on our host.
            self.logger.debug(f"Port {port} is already forwarded on this host.")
            return

        try:
            self.logger.info(f"Checking port {port}.")
            socket.socket().bind(("localhost", port))
        except OSError:
            # Port is already in use.
            if port in self.all_forwarded_ports:
                # TODO: In addition to terminating ourselves, also terminate other
                # workers on our node (also the dask-worker process!).
                self.abort(
                    multiline(
                        f"""
                        Port {port} from {worker_id} was forwarded on other hosts (but
                        not ours), but is unavailable on our host. Since we cannot
                        change the port (it is already forwarded) we must retire
                        workers on our host as connection to the cluster is now
                        impossible. Exit (1)."""
                    )
                )

            self.logger.warning(f"Port {port} from {worker_id} is unavailable.")
            # Note that if the port has been changed since (another process
            # marked the port as invalid before and it has been changed as a
            # response) this will have no effect due to the 'WHERE' clause.
            with self.con:
                self.con.execute(
                    "UPDATE workers "
                    "SET invalid_port = True "
                    "WHERE port = ? OR nanny_port = ?",
                    (port,),
                )

    def output(self):
        """Handle outputting worker port and issuing SSH commands.

        Race conditions should be handled by database triggers where applicable.

        """
        if self.is_scheduler:
            return

        ssh_command = self.remote_forwarding() + self.local_forwarding()

        if ssh_command:
            # If there is any forwarding to carry out.

            # Add any options and the remote hostname that ports will be forwarded to.
            ssh_command = (
                ("ssh",) + self.ssh_opts + ssh_command + (self.scheduler_hostname,)
            )
            self.logger.info(f"Running ssh command: {ssh_command}.")
            self.ssh_procs.append(Popen(ssh_command))
        else:
            self.logger.debug("No new ports to forward.")

    def remote_forwarding(self):
        """Create remote-forwarding command if needed.

        Returns:
            tuple: Remote-forwarding command.

        """
        ssh_command = ()
        if self.hostname == self.scheduler_hostname:
            # If we are on the same host as the scheduler, port forwarding would be
            # counterproductive.
            return ssh_command

        if not all(
            self.con.execute(
                "SELECT EXISTS(SELECT * FROM remote_forwards WHERE port = ?)",
                (self.port,),
            )
        ):
            # Our port has not been remote-forwarded (or output) yet.

            if self.nanny and all(
                self.con.execute(
                    "SELECT EXISTS(SELECT * FROM remote_forwards WHERE nanny_port = ?)",
                    (self.nanny_port,),
                )
            ):
                # Our nanny port has already been forwarded.
                self.abort(
                    "While our worker part was not forwarded, our nanny port was."
                )

            if self.output_already:
                raise RemoteForwardOutputError(
                    "Our port has been remote-forwarded already, but not output."
                )
            # Register this new remote forward in the database.
            with self.con:
                self.con.execute(
                    f"INSERT INTO remote_forwards VALUES (?, ?, ?)",
                    (self.name, self.port, self.nanny_port),
                )
            # Create the remote forwarding command.
            ssh_command += ("-R " + ":".join((f"localhost:{self.port}",) * 2),)
            if self.nanny:
                ssh_command += (
                    "-R " + ":".join((f"localhost:{self.nanny_port}",) * 2),
                )
            if not self.nanny:
                # Output worker port.
                self.logger.info(
                    f"Writing worker port {self.port} to file {self.output_file}."
                )
                if self.debug:
                    print("Port:", self.port)
                else:
                    with open(self.output_file, "w") as f:
                        f.write(f"{self.port}\n")
            else:
                # Output worker and nanny ports.
                self.logger.info(
                    f"Writing worker port {self.port} and nanny port "
                    f"{self.nanny_port} to file {self.output_file}."
                )
                if self.debug:
                    print("Ports:", self.port, self.nanny_port)
                else:
                    with open(self.output_file, "w") as f:
                        f.write(f"{self.port} {self.nanny_port}\n")

            self.output_already = True

        return ssh_command

    def local_forwarding(self):
        """Create local-forwarding command if needed.

        Returns:
            tuple: Local-forwarding command.

        """
        ssh_command = ()

        if self.hostname == self.scheduler_hostname:
            # If we are on the same host as the scheduler, port forwarding would be
            # counterproductive.
            return ssh_command

        # Determine ports which belong to workers on other nodes (but never the
        # scheduler, since it doesn't have a real port). Ports are guaranteed to be
        # unique due to the uniqueness constraint on the port column.
        ext_ports = set(
            self.con.execute(
                "SELECT port FROM workers WHERE hostname != ? AND name != 'scheduler'",
                (self.hostname,),
            )
        )

        # The foreign key constraints ensure that all ports (and worker names) found
        # in the 'local_forwards' and 'remote_forwards' tables are already recorded in
        # the 'workers' table.
        existing_local_forwards = set(
            self.con.execute(
                """
                SELECT local_forwards.port
                FROM local_forwards INNER JOIN workers USING(name)
                WHERE workers.hostname = ?""",
                (self.hostname,),
            )
        )

        assert ext_ports.issuperset(
            existing_local_forwards
        ), "All local forwards should be associated with workers on other nodes."

        # Which of those ports have not been forwarded yet on our host.
        new_ports = ext_ports.difference(existing_local_forwards)
        for new_port in new_ports:
            # There are matching ports that have not been locally forwarded yet.
            try:
                # Register the new local forward in the database.
                with self.con:
                    self.con.execute(
                        "INSERT INTO local_forwards VALUES (?, ?)",
                        (self.name, new_port),
                    )
                # Create the corresponding local forwarding command.
                ssh_command += ("-L " + ":".join((f"localhost:{new_port}",) * 2),)
            except IntegrityError as integrity_error:
                # This could happen if another process on the same node was already
                # carrying out the identical transaction at the time we requested the
                # transaction. If unchecked, this would lead to identical SSH
                # forwarding commands being issued, which the TRIGGER avoids.
                # Local forwarding happens on a first come, first served
                # basis and therefore has to be coordinated like this (or only one
                # worker per host could be made responsible for it).
                self.logger.error(
                    f"Possible race condition intercepted: {integrity_error}"
                )
        return ssh_command

    def increment_counter(self):
        """Increment our keepalive-counter."""
        with self.con:
            # Fetch counter.
            current_counter = self.con.execute(
                "SELECT counter FROM workers WHERE name = ?", (self.name,)
            ).fetchone()
            # Increment and write back.
            self.con.execute(
                "UPDATE workers SET counter = ? WHERE name = ?",
                (current_counter + 1, self.name),
            )

    @property
    def locked_port(self):
        """If our port is being forwarded, then we are no longer able to change it."""
        if not self._locked_port:
            if self.timeout_exceeded():
                # Optimisation - we save a query if the timeout has not expired yet.
                self._locked_port = self.port in self.all_forwarded_ports
        return self._locked_port

    def port_is_locked(self, port=None):
        """Check if `port` is locked.

        If `port` is being forwarded, we are no longer able to change it.

        Args:
            port (int): Port to check. If None (default), check our own port.

        Returns:
            bool: Whether the port is locked.

        """
        if port is None:
            return self.locked_port
        if self.timeout_exceeded():
            # Optimisation - we save a query if the timeout has not expired yet.
            return port in self.all_forwarded_ports
        return False

    def get_new_port(self):
        """Select a different port."""
        for _ in range(100):
            # Try for 100 times at most to set a new, unique port.
            try:
                with self.con:
                    self.con.execute(
                        """
                        UPDATE workers
                        SET port = ?,{' nanny_port = ?,' if self.nanny else ''}
                            invalid_port = False
                        WHERE name = ?""",
                        (
                            get_ports()[0],
                            *((get_ports()[0],) if self.nanny else ()),
                            self.name,
                        ),
                    )
                    # Notify other workers that they need to re-check ports.
                    self.notify()
                break  # Stop trying to get a new port.
            except IntegrityError as integrity_error:
                if re.match(
                    (
                        "(^Port already exists \(worker port or nanny port\).$"
                        "|^UNIQUE constraint failed: workers\..*$)"
                    ),
                    str(integrity_error),
                ):
                    self.logger.debug("Encountered port number clash.")
                else:
                    # Do not intercept other errors.
                    raise
        else:
            # No break encountered - the new port was always already present.
            self.abort("Could not find a suitable unique port. Exit (1).")

    @property
    def all_forwarded_ports(self):
        """All ports that are being forwarded across all workers.

        Returns:
            set: All (globally) forwarded ports.

        """
        return set(
            list(self.con.execute("SELECT port FROM local_forwards"))
            + list(self.con.execute("SELECT port FROM remote_forwards"))
            + list(self.con.execute("SELECT nanny_port FROM remote_forwards"))
        )

    @property
    def host_forwarded_ports(self):
        """The ports that are being forwarded on our host.

        If we are the scheduler, then this is equivalent to `self.all_forwarded_ports`
        since all ports are forwarded to the scheduler.

        """
        if self.hostname == self.scheduler_hostname:
            return self.all_forwarded_ports
        return set(
            reduce(
                add,
                (
                    list(
                        self.con.execute(
                            f"""
                            SELECT {forward_tab}.{column}
                            FROM {forward_tab} INNER JOIN workers USING(name)
                            WHERE workers.hostname = ?""",
                            (self.hostname,),
                        )
                    )
                    for forward_tab, column in zip(
                        ("local_forwards", *("remote_forwards",) * 2),
                        (*("port",) * 2, "nanny_port"),
                    )
                ),
            )
        )

    @property
    def scheduler_hostname(self):
        """The remote hostname that ports are forwarded to."""
        return self.con.execute(
            "SELECT hostname FROM workers WHERE name = 'scheduler'"
        ).fetchone()

    @property
    def n_workers(self):
        return self.con.execute("SELECT COUNT(*) FROM workers").fetchone()

    def any_invalid_ports(self):
        return any(self.con.execute("SELECT invalid_port FROM workers"))

    def port_is_invalid(self, port=None):
        """Check if port is invalid.

        Args:
            port (int): Port to check. If None (default), check our own port.

        Returns:
            bool: Whether the port is invalid.

        """
        if port is None:
            port = self.port
        return any(
            self.con.execute("SELECT invalid_port FROM workers WHERE port = ?", (port,))
        )

    def notify(self, hostname=None):
        """Notify other workers that our state has changed by resetting everyone."""
        with self.con:
            if hostname is None:
                self.logger.info("Notifying other workers.")
                self.con.execute("UPDATE workers SET ready = False")
            else:
                self.logger.info(f"Notifying other workers on {hostname}.")
                self.con.execute(
                    "UPDATE workers SET ready = False WHERE hostname = ?", (hostname,)
                )
        self.logger.debug("Finished notifying other workers.")

    def timeout_exceeded(self):
        if self.con.execute("SELECT exceeded FROM timeout_exceeded").fetchone() is None:
            if (time() - self.start_time) > self.initial_timeout:
                # We have now surpassed the initial timeout, so we need to write
                # this back to the database.
                with self.con:
                    self.con.execute("INSERT INTO timeout_exceeded VALUES (True)")
                return True
            return False
        return True

    def abort(self, msg=None, code=1):
        """Abort and signal the error to the job submission script."""
        self.logger.critical(f"Exit ({code})." if msg is None else msg)
        if self.debug:
            print("Port:", -1)
        else:
            with open(self.output_file, "w") as f:
                f.write("-1\n")
        self.close()
        sys.exit(code)

    def close(self):
        """Clean up."""
        # If called explicitly, do not clean up later.
        atexit.unregister(self.close)

        # Kill SSH processes first.
        self._kill_ssh_procs()

        # After SSH processes are killed, remote our entry from the database,
        # prompting other workers on our host to take over port forwarding if needed.
        self.clear()

        # Cancel the repeated execution of the synchronisation loop.
        self.loop_scheduler.cancel()

        # Close the database connection last.
        self.con.close()

    def _kill_ssh_procs(self):
        for proc in self.ssh_procs:
            self.logger.warning(f"Killing: {proc}.")
            proc.kill()

    def clear(self, name=None):
        """To be called when this worker exits upon encountering an error.

        Remove all rows relevant to the specified worker.
        If our own worker is cleared, also stop the synchronisation loop.

        Args:
            name (str): Name of the worker to clear. Defaults to `self.name`.

        """
        if name is None:
            name = self.name
            port = self.port
        else:
            port = self.con.execute(
                "SELECT port FROM workers WHERE name = ?", (name,)
            ).fetchone()
        self.logger.warning(f"Clearing worker {name} with port {port}.")
        with self.con:
            self.con.execute(
                f"DELETE FROM local_forwards WHERE name = ? OR port = ?", (name, port)
            )
            for table in ("remote_forwards", "workers"):
                self.con.execute(f"DELETE FROM {table} WHERE name = ?", (name,))

    def __str__(self):
        # Add general worker information.
        workers_columns = (
            "name",
            "hostname",
            "port",
            "ready",
            "invalid_port",
            "counter",
        )
        df_columns = workers_columns
        df_data = tuple(
            list(row)
            for row in self.con.execute(
                f"SELECT {' ,'.join(workers_columns)} FROM workers"
            )
        )
        # Add nanny port column.
        df_columns += ("nanny_port",)
        for row in df_data:
            nanny_port = self.con.execute(
                "SELECT nanny_port FROM workers WHERE name = ?", (row[0],)
            ).fetchone()
            row.append(nanny_port if nanny_port else 0)

        # Add remote forwarding information.
        df_columns += ("remote forward",)
        for row in df_data:
            remote_forward = self.con.execute(
                "SELECT port FROM remote_forwards WHERE name = ?", (row[0],)
            ).fetchone()
            row.append(remote_forward if remote_forward else 0)
        df_columns += ("remote forward nanny",)
        for row in df_data:
            nanny_remote_forward = self.con.execute(
                "SELECT nanny_port FROM remote_forwards WHERE name = ?", (row[0],)
            ).fetchone()
            row.append(nanny_remote_forward if nanny_remote_forward else 0)

        # Add local forwarding information.
        df_columns += ("local forwards",)

        local_forwards = defaultdict(list)
        for worker_id, local_forward in self.con.execute(
            """
            SELECT workers.name, local_forwards.port
            FROM workers
            LEFT JOIN local_forwards ON workers.name=local_forwards.name"""
        ):
            local_forwards[worker_id].append(local_forward)

        assert workers_columns[0] == "name"
        for row in df_data:
            row.append(local_forwards[row[0]])

        df = pd.DataFrame(df_data, columns=df_columns).sort_values(["hostname", "name"])
        return df.to_string(index=False)


def main():
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--data-file",
        default=port_sync_args["data_file"].default,
        help=port_sync_args["data_file"].help,
    )
    parser.add_argument(
        "-t",
        "--initial-timeout",
        type=float,
        default=port_sync_args["initial_timeout"].default,
        help=port_sync_args["initial_timeout"].help,
    )
    parser.add_argument(
        "-i",
        "--poll-interval",
        type=float,
        default=port_sync_args["poll_interval"].default,
        help=port_sync_args["poll_interval"].help,
    )
    parser.add_argument(
        "-s", "--scheduler", action="store_true", help=port_sync_args["scheduler"].help
    )
    parser.add_argument(
        "--debug", action="store_true", help=port_sync_args["debug"].help
    )
    parser.add_argument(
        "--output-file",
        nargs="?",
        default=port_sync_args["output_file"].default,
        help=port_sync_args["output_file"].help,
    )
    parser.add_argument(
        "--ssh-opts",
        nargs="?",
        default=" ".join(port_sync_args["ssh_opts"].default),
        help=port_sync_args["ssh_opts"].help,
    )
    parser.add_argument(
        "--hostname",
        nargs="?",
        default=port_sync_args["hostname"].default,
        help=port_sync_args["hostname"].help,
    )
    parser.add_argument(
        "--port",
        type=int,
        nargs="?",
        default=port_sync_args["port"].default,
        help=port_sync_args["port"].help,
    )
    parser.add_argument(
        "--keepalive-intervals",
        type=int,
        nargs="?",
        default=port_sync_args["keepalive_intervals"].default,
        help=port_sync_args["keepalive_intervals"].help,
    )
    parser.add_argument(
        "--nanny", action="store_true", help=port_sync_args["debug"].help
    )
    parser.add_argument(
        "--nanny-port",
        type=int,
        nargs="?",
        default=port_sync_args["nanny_port"].default,
        help=port_sync_args["nanny_port"].help,
    )
    args = parser.parse_args()

    enable_logging(level=logging.DEBUG if args.debug else logging.INFO)

    PortSync(
        args.data_file,
        args.initial_timeout,
        args.poll_interval,
        args.scheduler,
        args.debug,
        args.output_file,
        tuple(shlex.split(args.ssh_opts)),
        args.hostname,
        args.port,
        args.keepalive_intervals,
        args.nanny,
        args.nanny_port,
    ).run()


if __name__ == "__main__":
    main()
