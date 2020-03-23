#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Determine unused ports.

The number of ports can be specified.

"""
import socket
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_ports(n=1):
    """Get unused ports.

    We are opening sockets and binding them to ports that the OS deems to be
    available. By closing these sockets thereafter, we can be relatively certain
    that the previously determined ports will be available. We are also binding all
    sockets simultaneously to avoid duplicate port numbers.

    See https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port

    Args:
        n (int): The number of ports to retrieve.

    Returns:
        list of int: The ports.

    """
    ports = []
    sockets = []

    for i in range(n):
        sockets.append(socket.socket())
        sockets[-1].bind(("", 0))
        ports.append(sockets[-1].getsockname()[1])

    for s in sockets:
        s.close()

    return ports


def main():
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("n", default=1, type=int, nargs="?", help="the number of ports")
    args = parser.parse_args()

    print(" ".join(map(str, get_ports(args.n))))


if __name__ == "__main__":
    main()
