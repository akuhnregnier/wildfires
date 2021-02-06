# -*- coding: utf-8 -*-
import iris
import numpy as np
import pytest
from psutil import virtual_memory

from wildfires.data.datasets import data_is_available
from wildfires.utils import get_centres


def allequal(x, y):
    return np.all(x == y)


def available_memory_gb():
    return virtual_memory().total / 1024 ** 3


data_availability = pytest.mark.skipif(
    not data_is_available() or available_memory_gb() < 10,
    reason="Data directory is unavailable or there is not enough memory.",
)


def simple_cube(
    data,
    lat_lims=(-90, 90),
    lon_lims=(-180, 180),
    lat_points=None,
    lon_points=None,
    **kwargs
):
    """Generate a Cube from data and coordinates.

    Geographical coordinates can be given in terms of their limits or their points
    directly.

    """
    lat_dim = iris.coords.DimCoord(
        (
            get_centres(np.linspace(*lat_lims, data.shape[0] + 1))
            if lat_points is None
            else lat_points
        ),
        standard_name="latitude",
        units="degrees",
    )
    lon_dim = iris.coords.DimCoord(
        (
            get_centres(np.linspace(*lon_lims, data.shape[1] + 1))
            if lon_points is None
            else lon_points
        ),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    for coord in (lat_dim, lon_dim):
        coord.guess_bounds()

    return iris.cube.Cube(
        data,
        dim_coords_and_dims=[
            (
                lat_dim,
                0,
            ),
            (
                lon_dim,
                1,
            ),
        ],
        **kwargs,
    )
