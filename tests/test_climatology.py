# -*- coding: utf-8 -*-
import iris
import numpy as np
from numpy.testing import assert_allclose

from wildfires.data.datasets import MonthlyDataset, dummy_lat_lon_cube


class DummyDataset(MonthlyDataset):
    """Dataset with repeating (time axis) data."""

    def __init__(self):
        np.random.seed(1)
        repeating_data = np.random.random((12, 10, 10))
        data = np.vstack(tuple(repeating_data for i in range(2)))
        self.cubes = iris.cube.CubeList([dummy_lat_lon_cube(data, monthly=True)])


def test_climatology():
    # We know that the data is simply repeating in the DummyDataset.
    monthly = DummyDataset()
    climatology = monthly.get_climatology_dataset(monthly.min_time, monthly.max_time)

    shifted = monthly.get_temporally_shifted_dataset(months=-3, deep=False)
    shifted_climatology = shifted.get_climatology_dataset(
        shifted.min_time, shifted.max_time
    )

    assert_allclose(
        climatology.cube.data, np.roll(shifted_climatology.cube.data, shift=-3, axis=0)
    )
