# -*- coding: utf-8 -*-
"""Test the handling of masked source grid elements.

When masked elements are present in the source grid, they should not influence the
result.

In the case of linear interpolation, all adjacent points on the output grid should be
masked, while this depends on a threshold for area weighted regridding.

"""
import numpy as np
import pytest

from wildfires.data.datasets import dummy_lat_lon_cube, regrid
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

    assert np.allclose(interp_cube.data, target_data)


@pytest.mark.parametrize("mdtol", (0, 1))
def test_linear_masked_regrid(mdtol):
    source_data = np.ma.MaskedArray(
        [[1, 1, 1], [1, 1, 1], [2, 2, 100], [1, 1, 1], [1, 1, 1]],
        mask=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        dtype=np.float64,
    )
    source_cube = dummy_lat_lon_cube(source_data)

    lat_bounds = source_cube.coord("latitude").bounds
    row_weights = np.array(
        [
            np.sin(np.deg2rad(lats_u)) - np.sin(np.deg2rad(lats_l))
            for lats_l, lats_u in lat_bounds
        ]
    )

    # This is what we expect the output to be like.
    target_data = np.ma.MaskedArray(
        [
            [1, 1],
            [
                (2 * row_weights[2] + row_weights[1]) / np.sum(row_weights[1:3]),
                (2 * row_weights[2] + 2 * row_weights[1])
                / (row_weights[2] + 2 * row_weights[1]),
            ],
        ],
        mask=np.array([[0, 0], [0, 1 if mdtol < 1 else 0]]),
    )
    target_data = np.ma.vstack((target_data, target_data[::-1]))

    # Carry out the area weighted interpolation.
    interp_cube = regrid(
        source_cube,
        area_weighted=True,
        new_latitudes=get_centres(source_cube.coord("latitude").points),
        new_longitudes=get_centres(source_cube.coord("longitude").points),
        mdtol=mdtol,
    )

    assert np.allclose(interp_cube.data, target_data, atol=1e-1)
