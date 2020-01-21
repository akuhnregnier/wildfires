# -*- coding: utf-8 -*-
import iris
import numpy as np
import pytest

from wildfires.data.datasets import dummy_lat_lon_cube, regrid
from wildfires.utils import get_centres


@pytest.fixture
def source_data():
    return np.array([[3, 1, 1], [2, 2, 3], [1, 1, 1], [1, 3, 4]], dtype=np.float64)


@pytest.fixture(scope="session")
def area_weight_factors_4_2():
    factors = np.array(
        [
            np.sin((np.pi / 180) * 90) - np.sin((np.pi / 180) * 45),
            np.sin((np.pi / 180) * 45) - np.sin((np.pi / 180) * 0),
        ]
    )
    factors /= np.sum(factors)
    # Flip along after the equator due to symmetry.
    return np.array([*factors, *factors[::-1]]).reshape(4, 1)


@pytest.fixture()
def cubes_2D(source_data, area_weight_factors_4_2):
    target_data = (source_data * area_weight_factors_4_2).reshape(2, 2, 3).sum(axis=1)

    source_target_cubes = iris.cube.CubeList()
    for data in (source_data, target_data):
        source_target_cubes.append(dummy_lat_lon_cube(data))
    return source_target_cubes


@pytest.fixture
def cubes_3D(cubes_2D):
    source_target_cubes = iris.cube.CubeList()
    for cube in cubes_2D:
        data = np.vstack([cube.data[np.newaxis] for i in range(5)])
        source_target_cubes.append(dummy_lat_lon_cube(data))
    return source_target_cubes


def test_2D_regrid(cubes_2D):
    source_cube, target_cube = cubes_2D
    regridded = regrid(
        source_cube,
        area_weighted=True,
        new_latitudes=target_cube.coord("latitude"),
        new_longitudes=target_cube.coord("longitude"),
    )

    assert np.all(np.isclose(regridded.data, target_cube.data))


def test_return_identical_cube(cubes_2D):
    # NOTE: Does not apply for 3D data due to the current
    # implementation, but the 2D implementation is used recursively by
    # the 3D version.
    _, target_cube = cubes_2D
    regridded = regrid(
        target_cube,
        area_weighted=True,
        new_latitudes=target_cube.coord("latitude"),
        new_longitudes=target_cube.coord("longitude"),
    )

    assert id(regridded) == id(
        target_cube
    ), "(2D) The regridded and target cubes should be identical."


def test_3D_regrid(cubes_3D):
    time_source_cube, time_target_cube = cubes_3D
    time_regridded = regrid(
        time_source_cube,
        area_weighted=True,
        new_latitudes=time_target_cube.coord("latitude"),
        new_longitudes=time_target_cube.coord("longitude"),
    )

    assert np.all(np.isclose(time_regridded.data, time_target_cube.data))


def test_lon():
    source_data = np.array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]])

    target_interp = (
        (
            np.array(
                [[1, 2 / 3, 0, 0, 0], [0, 1 / 3, 1, 1 / 3, 0], [0, 0, 0, 2 / 3, 1]]
            )
            * (3 / 5)
        )
        .dot(source_data.T)
        .T
    )

    regrid_cube = regrid(
        dummy_lat_lon_cube(source_data),
        area_weighted=True,
        new_latitudes=get_centres(np.linspace(-90, 90, target_interp.shape[0] + 1)),
        new_longitudes=get_centres(np.linspace(-180, 180, target_interp.shape[1] + 1)),
    )
    assert np.all(np.isclose(target_interp, regrid_cube.data))
