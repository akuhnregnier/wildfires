# -*- coding: utf-8 -*-
"""Program that is called from several nodes to coordinate worker port numbers.

Note that the port and lock files are not removed automatically upon finishing, and so
should be cleaned up by a coordinating process (most likely the Cluster init).

"""
import atexit
import json
import logging
import os
import shlex
import socket
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import Popen
from time import sleep, time

import fasteners

from wildfires.logging_config import enable_logging

from .ports import get_ports

logger = logging.getLogger(__name__)


def format_ports(ports):
    if isinstance(ports, str):
        return [ports]
    elif isinstance(ports, int):
        return [str(ports)]
    else:
        return [str(port) for port in ports]


def handle_output(
    worker_ports,
    job_id,
    hostname,
    log_func,
    ssh_opts=("-NT",),
    debug=False,
    output_file=None,
):
    ssh_command = ()
    # The remote hostname that ports are forwarded to.
    scheduler_hostname = worker_ports["workers"]["scheduler"]["hostname"]

    if not hasattr(handle_output, "finished_output"):
        # Only output this information the first time around.
        handle_output.finished_output = True
        if output_file:
            log_func(
                f"Writing worker port: {worker_ports['workers'][job_id]['port']},"
                f"to file: {output_file}."
            )
            with open(output_file, "w") as f:
                f.write(str(worker_ports["workers"][job_id]["port"]) + "\n")
        else:
            log_func(
                f"Printing worker port: {worker_ports['workers'][job_id]['port']}."
            )
            print(worker_ports["workers"][job_id]["port"])

    if hostname != scheduler_hostname and (
        str(worker_ports["workers"][job_id]["port"])
        not in worker_ports["hostnames"][hostname]["remote_forwards"]
    ):
        # Our port has not been forwarded yet.
        # We are also not on the same host as the scheduler (in which case the port
        # forwarding would be counterproductive).

        # Create the remote forwarding command.
        ssh_command += (
            "-R "
            + ":".join((f"localhost:{worker_ports['workers'][job_id]['port']}",) * 2),
        )
        # Register this new remote forward.
        worker_ports["hostnames"][hostname]["remote_forwards"].append(
            str(worker_ports["workers"][job_id]["port"])
        )

    # Which ports belong to workers on other nodes.
    ext_ports = set(
        format_ports(
            [
                worker_data["port"]
                for worker_data in worker_ports["workers"].values()
                if worker_data["hostname"] != hostname and worker_data["port"] != -2
            ]
        )
    )

    # Which of those ports have not been forwarded yet on our host.
    assert ext_ports.issuperset(
        worker_ports["hostnames"][hostname]["local_forwards"]
    ), "All ports that are locally forwarded should be accounted for."
    new_ports = ext_ports.difference(
        worker_ports["hostnames"][hostname]["local_forwards"]
    )

    if hostname != scheduler_hostname and new_ports:
        # Create the local forwarding command(s).
        for port in new_ports:
            ssh_command += ("-L " + ":".join((f"localhost:{port}",) * 2),)
        # Also register the new local forward(s).
        worker_ports["hostnames"][hostname]["local_forwards"].extend(new_ports)

    if ssh_command:
        # Add any options.
        ssh_command = ("ssh",) + ssh_opts + ssh_command

        # Add the remote hostname that ports will be forwarded to.
        ssh_command += (scheduler_hostname,)

        ssh_command = shlex.split(" ".join(ssh_command))
        log_func(f"Running ssh command: {ssh_command}.")
        if debug:
            return print(ssh_command)
        return Popen(ssh_command)

    log_func("No new ports to forward.")


def reset_ready_flags(worker_ports):
    """Make sure that all workers are reset to a non-ready state.

    This is useful since when new data is added, we have to ensure that other
    workers look at this data.

    Once the timeout has expired, states should be locked in place, so flags will no
    longer be reset!

    """
    for worker_data in worker_ports["workers"].values():
        worker_data["ready"] = False


def check_ports(
    worker_ports, job_id, hostname, log_func, timeout_exceeded, output_file=None
):
    # The remote hostname that ports are forwarded to.
    scheduler_hostname = worker_ports["workers"]["scheduler"]["hostname"]

    if hostname == scheduler_hostname:
        host_forwarded_ports = set.union(
            *(
                set(host_data["local_forwards"] + host_data["remote_forwards"])
                for host_data in worker_ports["hostnames"].values()
            )
        )
    else:
        host_forwarded_ports = set(
            worker_ports["hostnames"][hostname]["local_forwards"]
            + worker_ports["hostnames"][hostname]["remote_forwards"]
        )

    # If we have already forwarded ports and are therefore unable to change these.
    if not timeout_exceeded(worker_ports):
        locked_state = False
    else:
        # If our port is being forwarded, then we are no longer able to change it.
        locked_state = worker_ports["workers"][job_id]["port"] in host_forwarded_ports

    if (
        any(
            worker_data["port"] == -1
            for worker_data in worker_ports["workers"].values()
        )
        and not locked_state
    ):
        # If any port was unavailable, and we are not in a locked state.
        log_func("Encountered invalid port(s).")
        if worker_ports["workers"][job_id]["port"] == -1:
            log_func("Our own port is invalid.")
            # If our own port is invalid, select a different port (it is very unlikely
            # to receive the same port twice from `get_ports()`).
            worker_ports["workers"][job_id]["port"] = get_ports()[0]

            # Notify other workers that all ports need to be re-checked.
            reset_ready_flags(worker_ports)
    else:
        # If all ports are available (on other nodes), check that they work for us.
        for worker, worker_data in worker_ports["workers"].items():
            port = worker_data["port"]
            if port == -2:
                # Skip the scheduler.
                continue
            elif str(port) in host_forwarded_ports:
                # Skip ports that have already been forwarded.
                continue
            try:
                log_func(f"Checking port {port}.")
                socket.socket().bind(("localhost", port))
            except OSError:
                # Port is already in use.

                if worker != job_id and str(port) in set(
                    worker_ports["hostnames"][worker_data["hostname"]]["local_forwards"]
                    + worker_ports["hostnames"][worker_data["hostname"]][
                        "remote_forwards"
                    ]
                ):
                    # The worker is not us, and the port in question is already being
                    # forwarded on that host and thus cannot be changed.
                    log_func(
                        f"Port {port} from {worker} was already in use. However, port "
                        "forwarding has already taken place, meaning we cannot "
                        "continue, as the necessary port forwarding necessary to join "
                        "the cluster cannot be performed on this node. Exit (1)"
                    )
                    if output_file:
                        log_func(f"Writing worker port: -1 to file: {output_file}.")
                        with open(output_file, "w") as f:
                            f.write("-1\n")
                    else:
                        log_func(f"Printing worker port: -1.")
                        print(-1)
                    sys.exit(1)

                log_func(
                    f"Port {port} from {worker} was already in use. Marking as invalid."
                )
                worker_data["port"] = -1
        if any(
            worker_data["port"] == -1
            for worker_data in worker_ports["workers"].values()
        ):
            # If any invalid ports were detected above, mark all
            # workers as not ready.
            reset_ready_flags(worker_ports)
        else:
            # Since everything went well above, all ports are available on this host.
            for worker_data in worker_ports["workers"].values():
                if worker_data["hostname"] == hostname:
                    worker_data["ready"] = True

    return worker_ports["workers"][job_id]["port"]


def sync_worker_ports(
    port_file=os.path.join(os.path.expanduser("~"), "worker_dir", "ports.json"),
    lock_file=os.path.join(os.path.expanduser("~"), "worker_dir", ".lock"),
    initial_timeout=300,
    poll_interval=10,
    scheduler=False,
    debug=False,
    output_file=None,
):
    """To be called from multiple workers to coordinate worker port numbers.

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

    Args:
        port_file (str): Path to the file used to store each worker's desired worker
            port and other information required for this process.
        lock_file (str): Path to the shared lock file, which will be used to
            coordinate access to `port_file`.
        initial_timeout (float): Timeout in seconds before the port synchronisation is
            concluded. After the timeout, any additional workers will be tied into the
            cluster if the existing ports are available on the new node and rejected
            otherwise. The shortest timeout wins.
        poll_interval (float): Interval in seconds between file read operations.
        scheduler (bool): If True, this process should be on the scheduler node. In a
            set of processes running this program, only one should be run with
            `scheduler=True`. The scheduler will not suggest any ports and only report
            whether the ports suggested by the workers (run with `scheduler=False`)
            are usable on the scheduler node.
        debug (bool): If True, only print out ssh commands instead of executing them.
        output_file (str): Path to the file where the worker port will be written to.
            If not supplied, the worker port will be printed to stdout instead.

    """
    start = time()

    def timeout_exceeded(worker_ports):
        if not worker_ports["timeout_exceeded"]:
            worker_ports["timeout_exceeded"] = (time() - start) > initial_timeout
        return worker_ports["timeout_exceeded"]

    if scheduler:
        job_id = "scheduler"
    else:
        # This will be the unique worker identifier.
        job_id = os.environ["PBS_JOBID"].split(".")[0]

    # The hostname is important as ports only need to be forwarded once per host,
    # since intra-host communication should be possible without port forwarding.
    hostname = socket.gethostname()

    def log_func(msg):
        """Log at the info level while including unique worker identification."""
        logger.info(f"{job_id}/{hostname}: {msg.rstrip('.')}.")

    def read_worker_ports():
        if os.path.isfile(port_file):
            with open(port_file) as f:
                return json.load(f)
        else:
            log_func(f"Port file {port_file} was not found (should only happen once).")
            return {"timeout_exceeded": False, "workers": {}, "hostnames": {}}

    def write_worker_ports(worker_ports):
        with open(port_file, "w") as f:
            json.dump(worker_ports, f, indent=2, sort_keys=True)

    log_func("Starting port number sync.")

    if not scheduler:
        own_proposed_port = get_ports()[0]
        log_func(f"Proposing port {own_proposed_port}.")
    else:
        own_proposed_port = -2

    ssh_procs = []

    def kill_procs(procs):
        for proc in procs:
            proc.kill()

    # Kill any ssh (forwarding) processes when the interpreter exits.
    atexit.register(kill_procs, ssh_procs)

    while True:
        # Wait for the next cycle, allowing other processes to access the file.
        sleep(poll_interval)

        # Read the file or create an empty dictionary if this is the first process to
        # ever acquire the lock.
        log_func("Waiting to acquire lock.")
        with fasteners.InterProcessLock(lock_file):
            # Make sure that our own info is present and wait.
            worker_ports = read_worker_ports()
            if job_id not in worker_ports["workers"]:
                log_func("Our entry is not present, adding it now.")
                # Add our own info.
                worker_ports["workers"][job_id] = {
                    "port": own_proposed_port,
                    "hostname": hostname,
                    "ready": False,
                }
                # Make sure other workers are aware of the change in information.
                reset_ready_flags(worker_ports)

                if hostname not in worker_ports["hostnames"]:
                    # Keep track of which forwards have been executed on each host.
                    worker_ports["hostnames"][hostname] = {
                        "remote_forwards": [],
                        "local_forwards": [],
                    }
            elif all(
                worker_data["ready"] for worker_data in worker_ports["workers"].values()
            ) and timeout_exceeded(worker_ports):
                # If every host has checked the ports and the initial sync timeout has
                # passed, we are ready to start issuing forwarding commands.
                assert (
                    own_proposed_port == worker_ports["workers"][job_id]["port"]
                ), "Our proposed port should match our stored port."

                log_func(
                    f"All {len(worker_ports['workers'])} workers are ready. "
                    f"Our port:{own_proposed_port}."
                )
                if not scheduler:
                    # Print ports in a format determined by `forwarding`.
                    ssh_proc = handle_output(
                        worker_ports,
                        job_id,
                        hostname,
                        log_func,
                        debug=debug,
                        output_file=output_file,
                    )
                    if ssh_proc is not None:
                        ssh_procs.append(ssh_proc)
            elif worker_ports["workers"][job_id]["ready"]:
                # We are 'ready', but other hosts are not.
                # Any worker on our host has already checked the ports locally.
                log_func("Our host is ready. Waiting for other hosts or timeout.")
            else:
                # If we are not ready, check the ports on our host.
                log_func("Checking ports.")
                own_proposed_port = check_ports(
                    worker_ports,
                    job_id,
                    hostname,
                    log_func,
                    timeout_exceeded,
                    output_file=output_file,
                )

            # Ensure integrity of the port numbers. These checks should always
            # succeed no matter the state of the program.
            all_ports = set(
                [
                    str(worker_data["port"])
                    for worker_data in worker_ports["workers"].values()
                    if worker_data["port"] != -2
                ]
            )

            forwarded_port_sets = [
                set(forward_data["local_forwards"] + forward_data["remote_forwards"])
                for forward_data in worker_ports["hostnames"].values()
            ]

            assert all(
                len(port_set) <= len(all_ports) for port_set in forwarded_port_sets
            ), "The number of forwarded ports should not exceed the number of workers."

            assert all(
                all_ports.issuperset(port_set) for port_set in forwarded_port_sets
            ), "Only existing port numbers should be forwarded."

            # NOTE: Checking the reverse (all forwarded ports should exist on the
            # workers) is troublesome, since the list of forwarded ports is due to
            # change as individual nodes initiate forwarding in turn.

            # Finally, write the updated information.
            write_worker_ports(worker_ports)


def main():
    enable_logging()
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
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
        default=300,
        type=float,
        help=(
            "Timeout in seconds before the port synchronisation is concluded. After "
            "the timeout, any additional workers will be added to the cluster if the "
            "existing ports are available on the new node, but rejected otherwise. "
            "The shortest timeout wins."
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
        "--debug",
        action="store_true",
        help="Only print out ssh commands instead of executing them.",
    )
    parser.add_argument(
        "--output-file",
        nargs="?",
        help=(
            "Path to the file where the worker port will be written to. If not "
            "supplied, the worker port will be printed to stdout instead."
        ),
    )

    args = parser.parse_args()

    sync_worker_ports(
        args.port_file,
        args.lock_file,
        args.timeout,
        args.poll_interval,
        args.scheduler,
        args.debug,
        args.output_file,
    )


if __name__ == "__main__":
    main()
