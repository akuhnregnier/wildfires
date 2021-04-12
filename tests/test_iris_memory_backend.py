#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cf_units
import iris
import iris.coord_categorisation
import numpy as np

from wildfires.data.datasets import dummy_lat_lon_cube

from .utils import *  # noqa


def test_iris_backend(iris_memory):
    @iris_memory.cache(ignore=["cube"])
    def dummy_process_func(cube=None, a=10):
        print("Calling func2")
        cube2 = cube.copy()
        cube2.long_name = "cube2"
        cube2.data += 10
        return iris.cube.CubeList([cube, cube2])

    cube = dummy_lat_lon_cube(
        np.random.random((1000, 2000)),
        long_name="cube1",
        units=cf_units.Unit("1"),
        var_name="cube1",
    )

    cube2 = cube.copy()
    cube2.long_name = "cube2"
    cube2.data += 10
    cube_list = dummy_process_func(cube=cube, a=12)

    s1 = sorted(cube_list, key=lambda cube: cube.name())
    s2 = sorted(iris.cube.CubeList([cube, cube2]), key=lambda cube: cube.name())
    assert all(np.all(cube1.data == cube2.data) for cube1, cube2 in zip(s1, s2))

    cube_list2 = dummy_process_func(a=12)
    s3 = sorted(cube_list2, key=lambda cube: cube.name())
    s4 = sorted(iris.cube.CubeList([cube, cube2]), key=lambda cube: cube.name())
    assert all(np.all(cube1.data == cube2.data) for cube1, cube2 in zip(s3, s4))
