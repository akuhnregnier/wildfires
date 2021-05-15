# -*- coding: utf-8 -*-
import iris
import numpy as np
from numpy.testing import assert_allclose

from wildfires.data.datasets import regrid
from wildfires.utils import get_centres


def test_multi_coord():
    """Spatial regridding should only affect the latitude and longitude coordinates."""
    # Define the source data.
    source_data = np.random.default_rng(0).random((2, 3, 40, 40))
    source_latitudes = iris.coords.DimCoord(
        get_centres(np.linspace(-90, 90, source_data.shape[-2] + 1)),
        standard_name="latitude",
        units="degrees",
    )
    source_longitudes = iris.coords.DimCoord(
        get_centres(np.linspace(-180, 180, source_data.shape[-1] + 1)),
        standard_name="longitude",
        units="degrees",
    )
    source_cube = iris.cube.Cube(
        source_data,
        long_name="var",
        dim_coords_and_dims=[
            (
                iris.coords.DimCoord(
                    list(range(source_data.shape[0])), long_name="b_dim"
                ),
                0,
            ),
            (
                iris.coords.DimCoord(
                    list(range(source_data.shape[1])), long_name="a_dim"
                ),
                1,
            ),
            (source_latitudes, 2),
            (source_longitudes, 3),
        ],
    )

    # Define the target data.
    target_data = source_data[:, :, ::2, ::2]
    target_latitudes = source_latitudes[::2]
    target_longitudes = source_longitudes[::2]

    # Carry out nearest-neighbour regridding.
    target_cube = regrid(
        source_cube,
        new_latitudes=target_latitudes,
        new_longitudes=target_longitudes,
        scheme=iris.analysis.Nearest(),
    )
    assert target_cube.shape == target_data.shape
    assert all(
        c0.name() == c1.name()
        for (c0, c1) in zip(source_cube.coords(), target_cube.coords())
    )
    assert_allclose(target_cube.data, target_data)
