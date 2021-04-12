# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from wildfires.cache import ProxyMemory, mark_dependency

parser = ArgumentParser()
parser.add_argument("--tmp-dir", required=True)
parser.add_argument("--first-run", action="store_true")
args = parser.parse_args()

memory = ProxyMemory(args.tmp_dir)


@memory.cache
@mark_dependency
def f0(x):
    return x + 1


@memory.cache(dependencies=(f0,))
@mark_dependency
def f1(x):
    return f0(x) - 1


if args.first_run:
    assert f0(0) == 1
    assert f1(0) == 0
else:
    f0.check_in_store(0)
    f1.check_in_store(0)
