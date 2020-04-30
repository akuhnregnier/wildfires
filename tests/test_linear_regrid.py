# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest

from wildfires.data.datasets import dummy_lat_lon_cube, regrid
from wildfires.utils import get_centres


def simple_downsampling(data, factor):
    """Simple interpolation by a factor which is a multiple of 2.

    Examples:
        >>> import numpy as np
        >>> source_data = np.array([[1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        >>> np.all(np.isclose(
        ...     simple_downsampling(source_data, 2),
        ...     np.array([[5 / 4, 9 / 4], [1, 1]])
        ... ))
        True
        >>> np.all(np.isclose(
        ...     simple_downsampling(source_data, 4),
        ...     np.array([1])
        ... ))
        True

    """
    assert math.log(
        factor, 2
    ).is_integer(), "Need a multiple of 2 to apply simple array averaging."
    assert (
        data.shape[0] / factor
    ).is_integer(), "Grid needs to be divisible by the factor."
    interpolated = np.zeros((int(data.shape[0] / factor), int(data.shape[1] / factor)))
    for lat_i in range(interpolated.shape[0]):
        lat_offset = int((factor / 2) - 1 + lat_i * factor)
        for lon_j in range(interpolated.shape[1]):
            lon_offset = int((factor / 2) - 1 + lon_j * factor)
            interpolated[lat_i, lon_j] = np.mean(
                data[lat_offset : lat_offset + 2, lon_offset : lon_offset + 2]
            )
    return interpolated


@pytest.mark.parametrize("nlats", [8, 100])
@pytest.mark.parametrize("factor", [2, 4])
def test_linear_downsampling(nlats, factor):
    """Test downsampling by different factors."""
    np.random.seed(1)
    source_data = np.random.random((nlats, nlats * 2))
    simple_interp = simple_downsampling(source_data, factor)
    regrid_cube = regrid(
        dummy_lat_lon_cube(source_data),
        area_weighted=False,
        new_latitudes=get_centres(
            np.linspace(-90, 90, (source_data.shape[0] // factor) + 1)
        ),
        new_longitudes=get_centres(
            np.linspace(-180, 180, (source_data.shape[1] // factor) + 1)
        ),
    )
    assert np.allclose(simple_interp, regrid_cube.data)


def test_non_central():
    source_cube = dummy_lat_lon_cube(np.array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]))

    target_longitudes = get_centres(np.linspace(-180, 180, 4))
    dx = np.mean(np.diff(source_cube.coord("longitude").points))
    target_interp = np.array(
        [
            [
                source_cube.data[row, 0]
                + (source_cube.data[row, 1] - source_cube.data[row, 0])
                * (target_longitudes[0] - source_cube.coord("longitude").points[0])
                / dx,
                source_cube.data[row, 2],
                source_cube.data[row, 3]
                + (source_cube.data[row, 4] - source_cube.data[row, 3])
                * (target_longitudes[2] - source_cube.coord("longitude").points[3])
                / dx,
            ]
            for row in range(2)
        ]
    )

    regrid_cube = regrid(
        source_cube,
        area_weighted=False,
        new_latitudes=get_centres(np.linspace(-90, 90, target_interp.shape[0] + 1)),
        new_longitudes=target_longitudes,
    )
    assert np.allclose(target_interp, regrid_cube.data)
