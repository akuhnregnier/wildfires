# -*- coding: utf-8 -*-
"""Test the handling of masked source grid elements.

When masked elements are present in the source grid, they should not influence the
result.

In the case of linear interpolation, all adjacent points on the output grid should be
masked, while this depends on a threshold for area weighted regridding.

"""
import iris
import numpy as np
import pytest
from numpy.testing import assert_allclose

from wildfires.data.datasets import dummy_lat_lon_cube, homogenise_cube_mask, regrid
from wildfires.utils import get_centres


def test_linear_masked_regrid():
    source_data = np.ma.MaskedArray(
        [[1, 1, 1], [1, 1, 1], [2, 2, 100], [1, 1, 1], [1, 1, 1]],
        mask=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        dtype=np.float64,
    )

    # This is what we expect the output to be like.
    target_data = np.ma.MaskedArray(
        [[1, 1], [1.5, np.nan], [1.5, np.nan], [1, 1]],
        mask=np.array([[0, 0], [0, 1], [0, 1], [0, 0]]),
    )

    source_cube = dummy_lat_lon_cube(source_data)

    # Carry out the linear interpolation.
    interp_cube = regrid(
        source_cube,
        area_weighted=False,
        new_latitudes=get_centres(source_cube.coord("latitude").points),
        new_longitudes=get_centres(source_cube.coord("longitude").points),
    )

    assert_allclose(interp_cube.data, target_data)


@pytest.mark.parametrize("mdtol", (0, 1))
def test_area_weighted_masked_regrid(mdtol):
    """Area weighted regridding from (5, 3) to (4, 2)."""
    source_data = np.ma.MaskedArray(
        [[1, 1, 1], [1, 1, 1], [2, 2, 100], [1, 1, 1], [1, 1, 1]],
        mask=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        dtype=np.float64,
    )
    source_cube = dummy_lat_lon_cube(source_data)

    def get_lat_weights(lats):
        weights = np.abs(np.diff(np.sin(np.deg2rad(lats))))
        return weights / np.sum(weights)

    masked_corner_weights = get_lat_weights([0, 18, 45]).reshape(2, 1)
    masked_corner_weights = np.hstack(
        (masked_corner_weights, masked_corner_weights * 2)
    )
    masked_corner_weights[0, 1] = 0
    masked_corner_weights /= np.sum(masked_corner_weights)

    # This is what we expect the output to be like.
    target_data = np.ma.MaskedArray(
        [
            [1, 1],
            [
                np.sum(np.array([2, 1]) * get_lat_weights([0, 18, 45])),
                np.sum(masked_corner_weights * source_data[2:4, 1:]),
            ],
        ],
        mask=np.array([[0, 0], [0, 1 if mdtol < 1 else 0]]),
    )
    target_data = np.ma.vstack((target_data, target_data[::-1]))

    # Carry out the area weighted interpolation.
    interp_cube = regrid(
        source_cube,
        new_latitudes=get_centres(np.linspace(-90, 90, 5)),
        new_longitudes=get_centres(np.linspace(-180, 180, 3)),
        scheme=iris.analysis.AreaWeighted(mdtol=mdtol),
    )

    assert_allclose(interp_cube.data, target_data, atol=1e-1)


def test_nearest_neighbour_regrid():
    np.random.seed(1)
    source_cube = homogenise_cube_mask(
        dummy_lat_lon_cube(np.random.random((3, 100, 100)))
    )
    source_cube.data.mask = source_cube.data.data > 0.5
    nn_cube = regrid(source_cube)

    inv_nn_cube = regrid(
        nn_cube,
        new_latitudes=source_cube.coord("latitude").points,
        new_longitudes=source_cube.coord("longitude").points,
    )
    assert_allclose(source_cube.data, inv_nn_cube.data, atol=1e-1)
    for agg in (iris.analysis.MIN, iris.analysis.MAX):
        assert_allclose(
            source_cube.collapsed(("latitude", "longitude"), agg).data,
            nn_cube.collapsed(("latitude", "longitude"), agg).data,
        )
