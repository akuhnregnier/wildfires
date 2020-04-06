#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import atexit
import shutil
from tempfile import mkdtemp

import cf_units
import iris
import iris.coord_categorisation
import numpy as np
from joblib import Memory

from wildfires.data.datasets import dummy_lat_lon_cube
from wildfires.joblib.iris_backend import register_backend

register_backend()

cache_dir = mkdtemp("joblib")
# Make sure the temporary cache directory gets removed after.
atexit.register(shutil.rmtree, cache_dir)
memory = Memory(location=cache_dir, backend="iris", verbose=100)
# memory = Memory(
#     location=os.environ.get("TMPDIR", "/tmp"), backend="local2", verbose=100
# )
# memory = Memory(location=os.environ.get("TMPDIR", "/tmp"), backend="local", verbose=100)


@memory.cache(ignore=["cube"])
def dummy_process_func(cube=None, a=10):
    print("Calling func2")
    cube2 = cube.copy()
    cube2.long_name = "cube2"
    cube2.data += 10
    return iris.cube.CubeList([cube, cube2])


def test_iris_backend():
    memory.clear()
    cube = dummy_lat_lon_cube(
        np.random.random((1000, 2000)),
        long_name="cube1",
        units=cf_units.Unit("1"),
        var_name="cube1",
    )

    cube2 = cube.copy()
    cube2.long_name = "cube2"
    cube2.data += 10
    print("Start calling")
    cube_list = dummy_process_func(cube=cube, a=12)

    s1 = sorted(cube_list, key=lambda cube: cube.name())
    s2 = sorted(iris.cube.CubeList([cube, cube2]), key=lambda cube: cube.name())
    assert all(np.all(cube1.data == cube2.data) for cube1, cube2 in zip(s1, s2))

    cube_list2 = dummy_process_func.call_and_shelve(a=12)
    s3 = sorted(cube_list2.get(), key=lambda cube: cube.name())
    s4 = sorted(iris.cube.CubeList([cube, cube2]), key=lambda cube: cube.name())
    assert all(np.all(cube1.data == cube2.data) for cube1, cube2 in zip(s3, s4))


# TODO: @data_availability
# TODO: def test_avoid_duplicated_caching():
# TODO: """Make sure that data is not being cached multiple times erroneously."""
# TODO: # Could be achieved by parsing through the metadata.json files in the cache
# directory, or by looking at the debug log files.
# TODO: Ideally, the location of iris_memory would be changed to a temporary directory
# for the test, the testing functions run twice, and checked that the number of cached
# files does not change between the first and the second run!
