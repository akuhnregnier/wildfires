# -*- coding: utf-8 -*-

import iris
import numpy as np
import pytest
from numpy.testing import assert_allclose

from wildfires.data.datasets import MonthlyDataset, dummy_lat_lon_cube, regrid
from wildfires.utils import get_centres

from .utils import simple_cube


class DummyDataset(MonthlyDataset):
    def __init__(self, shape=(10, 360, 720)):
        cube = dummy_lat_lon_cube(
            np.ma.MaskedArray(np.random.normal(scale=100, size=shape))
        )
        self.cubes = iris.cube.CubeList([cube])


@pytest.mark.parametrize(
    "scheme",
    (
        iris.analysis.Linear(extrapolation_mode="linear"),
        iris.analysis.AreaWeighted(mdtol=1),
    ),
)
def test_upsampling(scheme):
    """Upsampling followed by downsampling should yield the same data.

    The 'mask' extrapolation mode is not tested here because it masks the outside of
    the array when upsampling as expected, which leads to an entirely masked array
    when downsampling afterwards (again, as expected).

    The AreaWeighted scheme is only tested for a single `mdtol` value because there is
    no masked data in this test.

    """
    high_res_lat_cont_bnds = np.linspace(-90, 90, 5)
    high_res_lats = get_centres(high_res_lat_cont_bnds)
    low_res_lats = high_res_lat_cont_bnds[1::2]

    high_res_lon_cont_bnds = np.linspace(-180, 180, 5)
    high_res_lons = get_centres(high_res_lon_cont_bnds)
    low_res_lons = high_res_lon_cont_bnds[1::2]

    data = np.array([[1, 2], [3, 4]], dtype=np.float64)

    low_res_cube = simple_cube(
        data,
        lat_points=low_res_lats,
        lon_points=low_res_lons,
    )
    # Avoid circular longitude semantics.
    low_res_cube.coord("longitude").circular = False

    upsampled_cube = regrid(
        low_res_cube,
        new_latitudes=high_res_lats,
        new_longitudes=high_res_lons,
        scheme=scheme,
    )

    downsampled_cube = regrid(
        upsampled_cube,
        new_latitudes=low_res_lats,
        new_longitudes=low_res_lons,
        scheme=scheme,
    )
    assert_allclose(data, downsampled_cube.data)


@pytest.mark.parametrize(
    "scheme",
    (
        iris.analysis.Linear(extrapolation_mode="linear"),
        iris.analysis.Linear(extrapolation_mode="mask"),
        pytest.param(iris.analysis.AreaWeighted(mdtol=1), marks=pytest.mark.slow),
        pytest.param(iris.analysis.AreaWeighted(mdtol=0.5), marks=pytest.mark.slow),
        pytest.param(iris.analysis.AreaWeighted(mdtol=0), marks=pytest.mark.slow),
    ),
)
def test_masked_upsampling(scheme):
    """For certain data/masks, the Linear regridder yields very large values.

    This seems to occur mainly along the edges of the interpolation region, but it is
    not clear if this is always the case. Note that most, but not all of these values
    end up being masked.

    Here, we test the workaround for this implemented in the `wildfires.data.regrid()`
    function - masking all output points which have values that are outside of the
    input maxima.

    For the case in this test (masked input data), the Linear regridder results do not
    seem to depend on the extrapolation mode at all, as opposed to the test above
    (unmasked input data, fewer data points), where the upsampled (extrapolated)
    border points were indeed masked with the 'mask' extrapolation mode!!

    """
    np.random.seed(2)

    orig = DummyDataset(shape=(2, 360, 720))
    orig.homogenise_masks()

    orig.cube.data.mask = orig.cube.data > 100

    reg = regrid(orig.cube, scheme=scheme)

    # Find the time slices with values above the original maximum or below the
    # original minimum.

    reg_time_maxs = np.max(reg.data, axis=(1, 2))
    reg_time_mins = np.min(reg.data, axis=(1, 2))

    extreme_times = np.where((reg_time_maxs > 150) | (reg_time_mins < -500))[0]

    assert not np.any(extreme_times)

    assert reg.shape[0] == orig.cube.shape[0]
    assert np.all(np.array(reg.shape[1:]) == 2 * np.array(orig.cube.shape[1:]))
