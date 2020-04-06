# -*- coding: utf-8 -*-
import atexit
import os
from functools import partial
from sqlite3 import IntegrityError
from tempfile import mkstemp
from time import monotonic

import numpy as np
import pytest

from wildfires.sync_worker_ports import PortSync, Scheduler


@pytest.fixture(scope="function")
def get_port_sync():
    """Return a factory function for PortSync instances."""
    # Generate a distinct database file.
    database_fd, database_file = mkstemp(suffix=".db", prefix="port_db_")
    # Make sure the temporary database file gets closed and removed after.
    close_database_file = partial(os.close, database_fd)
    remove_database_file = partial(os.remove, database_file)
    atexit.register(close_database_file)
    atexit.register(remove_database_file)

    # This represents other workers (non-schedulers) in the system.
    workers_values = [
        ("1", "host1", 1234, False, False, 0),
        ("2", "host1", 1235, False, False, 0),
        ("3", "host2", 1236, False, False, 0),
    ]

    def create_instance(
        scheduler=True, timeout_exceeded=False, hostname="host3", port=1237
    ):
        if not scheduler:
            # Set a worker job id for the PortSync instance.
            os.environ["PBS_JOBID"] = "worker_test.pbs"
            # Since we are not a scheduler, add a scheduler to the system.
            workers_values.append(("scheduler", "host1", -2, False, False, 0))
        if scheduler:
            port = -2
        port_sync = PortSync(
            data_file=database_file,
            scheduler=scheduler,
            debug=True,
            hostname=hostname,
            port=port,
        )
        with port_sync.con:
            port_sync.con.executemany(
                "INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?)", workers_values
            )
            port_sync.con.execute(
                "INSERT INTO timeout_exceeded VALUES (?)", (timeout_exceeded,)
            )
        return port_sync

    yield create_instance
    # Clean up database file.
    close_database_file()
    atexit.unregister(close_database_file)
    remove_database_file()
    atexit.unregister(remove_database_file)


@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
def test_validate_existing_local_forwards(get_port_sync, timeout_exceeded, hostname):
    """Test the `validate_existing_local_forwards` trigger.

    The same local forward is attempted to be added twice (could happen when two
    workers on different hosts attempt this concurrently).

    """
    port_sync = get_port_sync(
        scheduler=False, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    # Worker port being forwarded.
    local_forward_port = 1234

    def local_forward():
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO local_forwards VALUES (?, ?)",
                (port_sync.name, local_forward_port),
            )

    # Register a new local forward in the database.
    local_forward()

    # Attempt to re-register the local forward in the database.
    with pytest.raises(
        IntegrityError, match="^Duplicate local port forwarding attempted.$"
    ):
        local_forward()


@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
def test_validate_existing_remote_forwards(get_port_sync, timeout_exceeded, hostname):
    """Test the `validate_existing_remote_forwards` trigger.

    The same remote forward is attempted to be added twice (could happen when two
    workers on different hosts attempt this concurrently).

    """
    port_sync = get_port_sync(
        scheduler=False, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    def remote_forward():
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO remote_forwards VALUES (?, ?)",
                (port_sync.name, port_sync.port),
            )

    # Register our remote forward in the database.
    remote_forward()

    # Attempt to re-register the remote forward in the database.
    with pytest.raises(
        IntegrityError, match="^Duplicate remote port forwarding attempted.$"
    ):
        remote_forward()


@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
@pytest.mark.parametrize("invalidate", [True, False])
def test_no_invalid_port_local_forward(
    get_port_sync, invalidate, timeout_exceeded, hostname
):
    """Test the `no_invalid_port_local_forward` trigger."""
    port_sync = get_port_sync(
        scheduler=False, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    local_forward_port = 1234

    if invalidate:
        # Invalidate the chosen port.
        with port_sync.con:
            port_sync.con.execute(
                "UPDATE workers SET invalid_port = True WHERE port = ?",
                (local_forward_port,),
            )

    def local_forward():
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO local_forwards VALUES (?, ?)",
                (port_sync.name, local_forward_port),
            )

    if invalidate:
        # Attempt to local-forward this port.
        with pytest.raises(
            IntegrityError, match="^Attempted local forward of invalid port.$"
        ):
            local_forward()
    else:
        # Should work fine.
        local_forward()


@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
@pytest.mark.parametrize("invalidate", [True, False])
def test_no_invalid_port_remote_forward(
    get_port_sync, invalidate, timeout_exceeded, hostname
):
    """Test the `no_invalid_port_remote_forward` trigger."""
    port_sync = get_port_sync(
        scheduler=False, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    if invalidate:
        # Invalidate our port.
        with port_sync.con:
            port_sync.con.execute(
                "UPDATE workers SET invalid_port = True WHERE port = ?",
                (port_sync.port,),
            )

    def remote_forward():
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO remote_forwards VALUES (?, ?)",
                (port_sync.name, port_sync.port),
            )

    if invalidate:
        # Attempt to remote-forward this port.
        with pytest.raises(
            IntegrityError, match="^Attempted remote forward of invalid port.$"
        ):
            remote_forward()
    else:
        # Should work fine.
        remote_forward()


@pytest.mark.parametrize("invalidate_port", [1234, 1235, 1236])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
@pytest.mark.parametrize("forward_ports", [(1234, 1235), (1235, 1236)])
@pytest.mark.parametrize("scheduler", [True, False])
def test_no_forwarded_port_invalidation(
    get_port_sync, scheduler, forward_ports, timeout_exceeded, hostname, invalidate_port
):
    """Test the `no_forwarded_port_invalidation` trigger.

    Once a port has been forwarded, any attempt to invalidate it should fail.

    """
    port_sync = get_port_sync(
        scheduler=scheduler, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    # Forward the specified ports.
    for port in forward_ports:
        if port == port_sync.port:
            # If it is our own port, we need to carry out remote forwarding.
            target_table = "remote_forwards"
        else:
            target_table = "local_forwards"

        with port_sync.con:
            port_sync.con.execute(
                f"INSERT INTO {target_table} VALUES (?, ?)", (port_sync.name, port)
            )

    def port_invalidation():
        # Invalidate the port.
        with port_sync.con:
            port_sync.con.execute(
                "UPDATE workers SET invalid_port = True WHERE port = ?",
                (invalidate_port,),
            )

    if invalidate_port in forward_ports:
        # This should result in an error.
        with pytest.raises(
            IntegrityError, match="^Attempted invalidation of forwarded port.$"
        ):
            port_invalidation()
    else:
        # Should work fine.
        port_invalidation()


@pytest.mark.parametrize("invalid", [True, False])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
@pytest.mark.parametrize("scheduler", [True, False])
def test_no_invalid_port_addition(
    get_port_sync, scheduler, timeout_exceeded, hostname, invalid
):
    """Test the `no_invalid_port_addition` trigger."""
    port_sync = get_port_sync(
        scheduler=scheduler, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    def new_worker():
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?)",
                ("test", hostname, port_sync.port + 1, False, invalid, 0),
            )

    if invalid:
        with pytest.raises(
            IntegrityError, match="^Attempted addition of invalid port.$"
        ):
            new_worker()
    else:
        new_worker()


def test_worker_removal(get_port_sync):
    port_sync = get_port_sync(scheduler=False, hostname="host2")
    with port_sync.con:
        port_sync.con.execute(
            "INSERT INTO remote_forwards VALUES (?, ?)",
            (port_sync.name, port_sync.port),
        )
        port_sync.con.executemany(
            "INSERT INTO local_forwards VALUES (?, ?)",
            ((port_sync.name, 1234), (port_sync.name, 1235)),
        )
    port_sync.clear()

    assert not list(
        port_sync.con.execute("SELECT * FROM workers WHERE name = ?", (port_sync.name,))
    )
    assert not list(
        port_sync.con.execute(
            "SELECT * FROM local_forwards WHERE name = ?", (port_sync.name,)
        )
    )
    assert not list(
        port_sync.con.execute(
            "SELECT * FROM remote_forwards WHERE name = ?", (port_sync.name,)
        )
    )
    assert port_sync.scheduler.scheduler.empty()


def test_scheduler():
    execution_times = []

    def action():
        execution_times.append(monotonic())

    interval = 1e-2
    max_iter = 10
    start = monotonic()
    # This is blocking.
    scheduler = Scheduler(interval, action, max_iter=max_iter).run()
    time_taken = monotonic() - start

    assert (
        scheduler.iterations == max_iter
    ), "We should be running the expected number of iterations."
    assert (
        scheduler.n_intervals == max_iter
    ), "Since our action takes a predictably small time we shouldn't skip intervals."
    assert (
        len(execution_times) == max_iter
    ), "The action should have been executed `max_iter` times."
    expected_total = (max_iter - 1) * interval
    assert (
        time_taken - expected_total
    ) / expected_total < 1e-2, (
        "The total time taken should be close to the expected time."
    )
    expected_times = execution_times[0] + (interval * np.arange(max_iter))
    deviation = np.asarray(execution_times) - expected_times
    assert (
        np.std(deviation) / interval < 0.1
    ), "The execution times should be relatively consistent."
