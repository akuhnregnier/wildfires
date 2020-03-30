#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import os
import socket
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from time import sleep, time

import fasteners

from wildfires.logging_config import enable_logging

from .ports import get_ports

logger = logging.getLogger(__name__)


def print_output(ports, forwarding=True):
    """Print forwarding instructions or space delimited port numbers."""
    if isinstance(ports, str):
        ports = [ports]
    elif isinstance(ports, int):
        ports = [str(ports)]
    else:
        ports = [str(port) for port in ports]

    if not forwarding:
        return print(" ".join(ports))

    output = (ports[0], f"-R localhost:{ports[0]}:localhost:{ports[0]}")
    for port in ports[1:]:
        output += (f"-L localhost:{port}:localhost:{port}",)
    print(" ".join(output))


def sync_worker_ports(
    n_workers,
    port_file=os.path.join(os.path.expanduser("~"), "worker_dir", "ports.json"),
    lock_file=os.path.join(os.path.expanduser("~"), "worker_dir", ".lock"),
    timeout=1200,
    poll_interval=10,
    scheduler=False,
    forwarding=True,
):
    """To be called from multiple workers to coordinate worker port numbers.

    Identical arguments must be used for each set of workers to coordinate.

    NOTE:

        This relies on a shared filesystem!

    Prints space delimited port numbers (integers) or forwarding instructions,
    depending on the value of `forwarding`.

    The first number is the local port number, which has to be forwarded to the
    scheduler via remote port forwarding.

    The other ports (if any) have to be forwarded to the scheduler via local port
    forwarding.

    The program will only print ports to be locally forwarded once per host.

    Args:
        n_workers (int): The number of workers to coordinate. This does not include
            the scheduler process.
        port_file (str): Path to the file used to store each worker's desired worker
            port and other information required for this process.
        lock_file (str): Path to the shared lock file, which will be used to
            coordinate access to `port_file`.
        timeout (float): Timeout in seconds before the port synchronisation is aborted.
            This should then lead to the job (and the whole cluster) failing since
            any worker failing this process makes it impossible to know which ports to
            forward.
        poll_interval (float): Interval in seconds between file read operations.
        scheduler (bool): If True, this process should be on the scheduler node. In a
            set of processes running this program, only one should be run with
            `scheduler=True`. The scheduler will not suggest any ports and only report
            whether the ports suggested by the workers (run with `scheduler=False`)
            are usable on the scheduler node.
        forwarding (bool): If True, print the local worker port and then the
            forwarding instructions separated by a space. Otherwise, simply print
            space delimited port numbers as outlined above.

    """
    if n_workers < 1:
        sys.exit(2)
    start = time()
    # The scheduler will have its own entry.
    n_expected = n_workers + 1
    if scheduler:
        job_id = "scheduler"
    else:
        # This will be the unique worker identifier.
        job_id = os.environ["PBS_JOBID"].split(".")[0]

    # The hostname is important as ports only need to be forwarded once per host,
    # since intra-host communication should be possible regardless.
    hostname = socket.gethostname()

    def log_func(msg):
        """Log at the info level while including unique worker identification."""
        logger.info(f"{job_id} -- {hostname}: {msg.rstrip('.')}.")

    def reset_ready_flags(worker_ports):
        """Make sure that all workers are reset to a non-ready state.

        This is useful since when new data is added, we have to ensure that other
        workers look at this data.

        """
        for worker_data in worker_ports.values():
            worker_data["ready"] = False

    log_func("Starting port number sync.")

    if not scheduler:
        own_proposed_port = get_ports()[0]
        log_func(f"Proposing port {own_proposed_port}.")
    else:
        own_proposed_port = -2

    while (time() - start) < timeout:
        # Read the file or create an empty dictionary if this is the first process to
        # ever acquire the lock.
        log_func("Waiting to acquire lock.")
        with fasteners.InterProcessLock(lock_file):
            if os.path.isfile(port_file):
                with open(port_file) as f:
                    worker_ports = json.load(f)
            else:
                worker_ports = {}
            n_present = len(worker_ports)
            if n_present < n_expected:
                # If fewer than `n_expected` entries have been added to the file, make
                # sure that our own info is present and wait.
                log_func(
                    f"Found {n_present} {'entries' if n_present != 1 else 'entry'}, "
                    "but expected {n_expected}."
                )
                # Make sure other workers are aware of the change in information.
                reset_ready_flags(worker_ports)
                # Add our own info.
                worker_ports[job_id] = {
                    "port": own_proposed_port,
                    "hostname": hostname,
                    "ready": False,
                    "exit": False,
                    "exited": False,
                }

            elif n_present == n_expected:
                log_func(f"Found {n_expected} entries as expected.")
                # If all workers have reported something, first check if another
                # worker on the same host as us has already checked the ports. In this
                # case, our own 'ready' status should have been set to True and we do
                # not need to do anything.
                if worker_ports[job_id]["ready"]:
                    if all(
                        worker_data["ready"] for worker_data in worker_ports.values()
                    ):
                        # If every process has checked the ports we are ready to
                        # return the results.
                        assert (
                            own_proposed_port == worker_ports[job_id]["port"]
                        ), "Our proposed port should match our stored port."
                        log_func(f"All host are ready. Our port:{own_proposed_port}.")
                        # To output our results, print a list of ports that need to be
                        # forwarded, ie. all ports which are not on our host.

                        if not scheduler:
                            if worker_ports[job_id]["exit"]:
                                # We have been told to exit by another worker on our
                                # host, who will also take care of the necessary port
                                # forwarding.
                                log_func("Told to exit (0).")
                                log_func(
                                    "Printing local port only as local port forwarding "
                                    "is handled by another worker."
                                )
                                print_output(own_proposed_port, forwarding=forwarding)
                            else:
                                # Tell other workers on the same host to exit after us.
                                for worker_data in worker_ports.values():
                                    if worker_data["hostname"] == hostname:
                                        worker_data["exit"] = True

                                log_func("Printing ports to forward.")
                                print_output(
                                    [str(own_proposed_port)]
                                    + [
                                        str(worker_data["port"])
                                        for worker_data in worker_ports.values()
                                        if worker_data["hostname"] != hostname
                                        and worker_data["port"] != -2
                                    ],
                                    forwarding=forwarding,
                                )

                        # Tell other workers that we have finished.
                        worker_ports[job_id]["exited"] = True
                        if all(
                            worker_data["exited"]
                            for worker_data in worker_ports.values()
                        ):
                            # Since everyone will have exited once we exit, delete the
                            # port file to prepare for the next cluster.
                            log_func(
                                "Every other worker has already exited. Deleting port "
                                "file."
                            )
                            os.remove(port_file)
                        else:
                            # Tell other workers who should exit, and that we have now
                            # finished.
                            with open(port_file, "w") as f:
                                json.dump(worker_ports, f, indent=2, sort_keys=True)

                        sys.exit(0)
                    log_func("Our host is ready. Waiting for other hosts.")
                else:
                    log_func("Checking ports.")
                    # Check for invalid ports first, which should be corrected before
                    # we proceed to check ports.
                    if any(
                        worker_data["port"] == -1
                        for worker_data in worker_ports.values()
                    ):
                        log_func("Encountered invalid port(s).")
                        if worker_ports[job_id]["port"] == -1:
                            assert (
                                not scheduler
                            ), "This should not happen if we are the scheduler."
                            log_func("Our own port is invalid.")
                            # If our own port is invalid, select a different port.
                            new_proposed_port = get_ports()[0]
                            assert (
                                own_proposed_port != new_proposed_port
                            ), "The new port should not be the same as the old one."
                            own_proposed_port = new_proposed_port

                            # Notify other workers and update our own port.
                            reset_ready_flags(worker_ports)
                            worker_ports[job_id]["port"] = own_proposed_port
                    else:
                        # Otherwise check that the given ports work for us.
                        for worker, worker_data in worker_ports.items():
                            port = worker_data["port"]
                            if port == -2:
                                # Skip the scheduler.
                                continue
                            try:
                                log_func(f"Checking port {port}.")
                                socket.socket().bind(("localhost", port))
                            except OSError:
                                # Port is already in use.
                                log_func(
                                    f"Port {port} from {worker} was already in use. "
                                    "Marking as invalid."
                                )
                                worker_data["port"] = -1
                        if any(
                            worker_data["port"] == -1
                            for worker_data in worker_ports.values()
                        ):
                            # If any invalid ports were detected above, mark all
                            # workers as not ready.
                            reset_ready_flags(worker_ports)
                        else:
                            # If everything went well above, that means that all ports
                            # work for us (and any other worker on the same host).
                            for worker_data in worker_ports.values():
                                if worker_data["hostname"] == hostname:
                                    worker_data["ready"] = True
            else:
                log_func(
                    f"Found {n_present} entries, but expected {n_expected} at most. "
                    "Exiting with code 3."
                )
                sys.exit(3)

            # Finally, write the updated information.
            with open(port_file, "w") as f:
                json.dump(worker_ports, f, indent=2, sort_keys=True)

        # Wait for the next poll.
        sleep(poll_interval)

    log_func("Timeout exceeded. Exiting with code 1.")
    sys.exit(1)


def main():
    enable_logging()
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help=(
            "The number of workers to coordinate. This does not include the scheduler "
            "process."
        ),
    )
    parser.add_argument(
        "-p",
        "--port-file",
        default=os.path.join(os.path.expanduser("~"), "worker_dir", "ports.json"),
        help=(
            "Path to the file used to store each worker's desired worker port and "
            "other information required for this process."
        ),
    )
    parser.add_argument(
        "-l",
        "--lock-file",
        default=os.path.join(os.path.expanduser("~"), "worker_dir", ".lock"),
        help=(
            "Path to the shared lock file, which will be used to coordinate access "
            "to `port_file`."
        ),
    )
    parser.add_argument(
        "-t",
        "--timeout",
        default=1200,
        type=float,
        help=(
            "Timeout in seconds before the port synchronisation is aborted. This should "
            "then lead to the job (and the whole cluster) failing since any worker failing "
            "this process makes it impossible to know which ports to forward."
        ),
    )
    parser.add_argument(
        "-i",
        "--poll-interval",
        default=10,
        type=float,
        help="Interval in seconds between file read operations.",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        action="store_true",
        help="Identify this process as the scheduler.",
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Do not print forwarding instructions, but space delimited port numbers.",
    )

    args = parser.parse_args()

    sync_worker_ports(
        args.workers,
        args.port_file,
        args.lock_file,
        args.timeout,
        args.poll_interval,
        args.scheduler,
        not args.raw,
    )


if __name__ == "__main__":
    main()
