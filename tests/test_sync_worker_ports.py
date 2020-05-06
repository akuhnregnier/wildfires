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

    def create_instance(
        scheduler=True,
        timeout_exceeded=False,
        hostname="host3",
        port=1237,
        nanny=False,
        nanny_port=None,
    ):
        # This represents other workers (non-schedulers) in the system.
        workers_values = [
            ("1", "host1", 1234, False, False, None if not nanny else 2234),
            ("2", "host1", 1235, False, False, None if not nanny else 2235),
            ("3", "host2", 1236, False, False, None if not nanny else 2236),
        ]
        if not scheduler:
            # Set a worker job id for the PortSync instance.
            os.environ["PBS_JOBID"] = "worker_test.pbs"
            # Since we are not a scheduler, add a scheduler to the system.
            workers_values.append(("scheduler", "host1", -2, False, False, None))

        port_sync = PortSync(
            data_file=database_file,
            scheduler=scheduler,
            debug=True,
            hostname=hostname,
            port=port,
            nanny=nanny,
            nanny_port=nanny_port,
            sqlite_timeout=0.2,
        )

        with port_sync.con:
            port_sync.con.executemany(
                "INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?)", workers_values
            )
            port_sync.con.execute(
                "INSERT INTO timeout_exceeded VALUES (?)", (timeout_exceeded,)
            )
        create_instance.port_sync = port_sync
        return port_sync

    yield create_instance
    # Stop running PortSync.
    create_instance.port_sync.close()
    # Clean up database file.
    close_database_file()
    atexit.unregister(close_database_file)
    remove_database_file()
    atexit.unregister(remove_database_file)


@pytest.mark.parametrize("nanny", [True, False])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
def test_duplicated_ports(get_port_sync, timeout_exceeded, hostname, nanny):
    """Ports and nanny ports should be unique without overlap."""
    if nanny:
        nanny_port = 2237
    else:
        nanny_port = None

    port_sync = get_port_sync(
        scheduler=False,
        timeout_exceeded=timeout_exceeded,
        hostname=hostname,
        nanny=nanny,
        nanny_port=nanny_port,
    )

    def insert_values(*values):
        port_sync.con.execute("INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?)", values)

    def update_ports(*, port=None, nanny_port=None, name=port_sync.name):
        if port is not None:
            port_sync.con.execute(
                "UPDATE workers SET port = ? WHERE name = ?", (port, name)
            )
        if nanny_port is not None:
            port_sync.con.execute(
                "UPDATE workers SET nanny_port = ? WHERE name = ?", (nanny_port, name)
            )

    def should_fail(mod_fn, args, kwargs):
        with pytest.raises(
            IntegrityError,
            match=(
                "(^Port already exists \(worker port or nanny port\).$"
                "|^UNIQUE constraint failed: workers\..*$)"
            ),
        ):
            with port_sync.con:
                mod_fn(*args, **kwargs)

    class IntentionalRollbackError(Exception):
        pass

    def should_pass(mod_fn, args, kwargs):
        with pytest.raises(IntentionalRollbackError):
            with port_sync.con:
                mod_fn(*args, **kwargs)
                raise IntentionalRollbackError()

    fail_tests = []
    fail_tests.append((insert_values, ("4", "host1", 1234, False, False, None), {}))
    if nanny:
        fail_tests.append((insert_values, ("4", "host1", 1234, False, False, 2234), {}))
        fail_tests.append((insert_values, ("4", "host1", 2222, False, False, 2234), {}))

    fail_tests.append((update_ports, (), dict(port=1234)))
    fail_tests.append((update_ports, (), dict(port=1236)))
    fail_tests.append((update_ports, (), dict(port=port_sync.port, name="1")))

    if nanny:
        fail_tests.append((update_ports, (), dict(port=2234)))
        fail_tests.append((update_ports, (), dict(port=2236)))
        fail_tests.append((update_ports, (), dict(nanny_port=2236)))
        fail_tests.append((update_ports, (), dict(nanny_port=1234)))

    for mod_fn, args, kwargs in fail_tests:
        should_fail(mod_fn, args, kwargs)

    pass_tests = []
    pass_tests.append((insert_values, ("4", "host1", 2222, False, False, 3333), {}))

    if not nanny:
        pass_tests.append((insert_values, ("4", "host1", 2222, False, False, 2234), {}))

    pass_tests.append((update_ports, (), dict(port=4444, nanny_port=5555)))
    pass_tests.append((update_ports, (), dict(port=4445, nanny_port=5556, name="2")))

    if not nanny:
        pass_tests.append((update_ports, (), dict(nanny_port=2234)))

    for mod_fn, args, kwargs in pass_tests:
        should_pass(mod_fn, args, kwargs)


@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
def test_duplicated_local_forwards(get_port_sync, timeout_exceeded, hostname):
    """Test that no local forwards are duplicated on the same host.

    The same local forward is attempted to be added twice (could happen when two
    workers on different hosts attempt this concurrently).

    """
    port_sync = get_port_sync(
        scheduler=False, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    # Worker port being forwarded.
    if hostname == "host1":
        local_forward_port = 1236
    elif hostname == "host2":
        local_forward_port = 1234

    def local_forward(worker_name=port_sync.name, forward_port=local_forward_port):
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO local_forwards VALUES (?, ?)", (worker_name, forward_port)
            )

    # Register a new local forward in the database.
    local_forward()

    # Attempt to re-register the local forward in the database.
    with pytest.raises(
        IntegrityError, match="^Duplicate local port forwarding attempted.$"
    ):
        local_forward()

    # Another worker on our host attempts to register the local forward.
    with pytest.raises(
        IntegrityError, match="^Duplicate local port forwarding attempted.$"
    ):
        local_forward(worker_name="1" if hostname == "host1" else "3")

    # A worker on another host attempts to register the local forward.
    local_forward(worker_name="3" if hostname == "host1" else "1")


@pytest.mark.parametrize("nanny", [True, False])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
def test_exclusive_remote_forwards(get_port_sync, timeout_exceeded, hostname, nanny):
    """Workers should only forward their own port(s) once."""
    if nanny:
        nanny_port = 2237
    else:
        nanny_port = None

    port_sync = get_port_sync(
        scheduler=False,
        timeout_exceeded=timeout_exceeded,
        hostname=hostname,
        nanny=nanny,
        nanny_port=nanny_port,
    )

    def remote_forward(worker_name=port_sync.name):
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO remote_forwards VALUES (?, ?, ?)",
                (worker_name, port_sync.port, port_sync.nanny_port),
            )

    # Register our remote forward in the database.
    remote_forward()

    # Attempt to re-register the remote forward in the database.
    with pytest.raises(
        IntegrityError,
        match="^UNIQUE constraint failed: remote_forwards\.(name|port|nanny_port)$",
    ):
        remote_forward()

    # Let another worker attempt to remote-forward our port.
    with pytest.raises(
        IntegrityError, match="^Attempted remote forward of foreign port.$"
    ):
        remote_forward(worker_name="1")


@pytest.mark.parametrize("nanny", [True, False])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
def test_nanny_port(get_port_sync, timeout_exceeded, hostname, nanny):
    """Nanny port should be None if `not nanny`."""
    if nanny:
        nanny_port = 2237
    else:
        nanny_port = None

    port_sync = get_port_sync(
        scheduler=False,
        timeout_exceeded=timeout_exceeded,
        hostname=hostname,
        nanny=nanny,
        nanny_port=nanny_port,
    )
    if nanny:
        assert port_sync.nanny_port == 2237
    else:
        assert port_sync.nanny_port is None


@pytest.mark.parametrize("nanny", [True, False])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
@pytest.mark.parametrize("invalidate", [True, False])
def test_invalid_port_local_forward(
    get_port_sync, invalidate, timeout_exceeded, hostname, nanny
):
    """Ensure that no invalid ports are local-forwarded."""
    if nanny:
        nanny_port = 2237
    else:
        nanny_port = None

    port_sync = get_port_sync(
        scheduler=False,
        timeout_exceeded=timeout_exceeded,
        hostname=hostname,
        nanny=nanny,
        nanny_port=nanny_port,
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


@pytest.mark.parametrize("nanny", [True, False])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
@pytest.mark.parametrize("invalidate", [True, False])
def test_invalid_port_remote_forward(
    get_port_sync, invalidate, timeout_exceeded, hostname, nanny
):
    """Ensure that no invalid ports are remote-forwarded."""
    if nanny:
        nanny_port = 2237
    else:
        nanny_port = None

    port_sync = get_port_sync(
        scheduler=False,
        timeout_exceeded=timeout_exceeded,
        hostname=hostname,
        nanny=nanny,
        nanny_port=nanny_port,
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
                "INSERT INTO remote_forwards VALUES (?, ?, ?)",
                (port_sync.name, port_sync.port, port_sync.nanny_port),
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


@pytest.mark.parametrize("nanny", [True, False])
@pytest.mark.parametrize("invalidate_port", [1234, 1235, 1236])
@pytest.mark.parametrize("hostname", ["host1", "host2"])
@pytest.mark.parametrize("timeout_exceeded", [True, False])
@pytest.mark.parametrize("forward_ports", [(1234, 1235), (1235, 1236)])
@pytest.mark.parametrize("scheduler", [True, False])
def test_forwarded_port_invalidation(
    get_port_sync,
    scheduler,
    forward_ports,
    timeout_exceeded,
    hostname,
    invalidate_port,
    nanny,
):
    """Once a port has been forwarded, any attempt to invalidate it should fail."""
    if nanny:
        nanny_port = 2237
    else:
        nanny_port = None

    port_sync = get_port_sync(
        scheduler=scheduler,
        timeout_exceeded=timeout_exceeded,
        hostname=hostname,
        nanny=nanny,
        nanny_port=nanny_port,
    )

    # Forward the specified ports.
    for port in forward_ports:
        with port_sync.con:
            if port == port_sync.port:
                # If it is our own port, we need to carry out remote forwarding.
                port_sync.con.execute(
                    f"INSERT INTO remote_forwards VALUES (?, ?, ?)",
                    (port_sync.name, port, port_sync.nanny_port),
                )
            else:
                port_sync.con.execute(
                    f"INSERT INTO local_forwards VALUES (?, ?)", (port_sync.name, port)
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
def test_invalid_port_addition(
    get_port_sync, scheduler, timeout_exceeded, hostname, invalid
):
    """No invalid ports may be added."""
    port_sync = get_port_sync(
        scheduler=scheduler, timeout_exceeded=timeout_exceeded, hostname=hostname
    )

    def new_worker():
        with port_sync.con:
            port_sync.con.execute(
                "INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?)",
                ("test", hostname, port_sync.port + 1, False, invalid, None),
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
            "INSERT INTO remote_forwards VALUES (?, ?, ?)",
            (port_sync.name, port_sync.port, port_sync.nanny_port),
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
    assert port_sync.loop_scheduler.scheduler.empty()


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
