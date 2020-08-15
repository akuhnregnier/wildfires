# -*- coding: utf-8 -*-

import iris
import numpy as np

from wildfires.data.datasets import MonthlyDataset, dummy_lat_lon_cube


class DummyDataset(MonthlyDataset):
    def __init__(self, data):
        self.cubes = iris.cube.CubeList([dummy_lat_lon_cube(data, monthly=True)])


def ma_comp(a, b):
    if np.all(a.data == b.data) and np.all(a.mask == b.mask):
        return True
    return False


def test_filling():
    """Filling of gaps using minimum values and a season-trend model."""
    # Generate 3 years of dummy data.
    data = np.zeros((36, 4, 4))
    year_data = np.array([0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0])
    data += np.repeat(year_data.reshape(1, 12), 3, axis=0).ravel().reshape(36, 1, 1)

    # Mask those elements that are at their minima, except for the very first ones.
    data = np.ma.MaskedArray(data, mask=data == 0)
    data.mask[0] = False

    # Modify data at one location.
    data[2:26:12, 0, 0] = 1

    # Fully mask one location - this should therefore remain unchanged.
    data.mask[:, 1, 1] = True

    orig = DummyDataset(data)
    filled = orig.get_persistent_season_trend_dataset()

    assert ma_comp(orig.cube.data[:, 1, 1], filled.cube.data[:, 1, 1])
    assert ma_comp(filled.cube.data[:, 1, 0], filled.cube.data[:, 0, 1])

    # Only one element should be altered.
    assert np.all(
        np.where(orig.cube.data.data[:, 0, 0] != filled.cube.data.data[:, 0, 0])[0]
        == np.array([26])
    )
    assert not np.any(filled.cube.data.mask[:, 0, 0])
    assert np.isclose(filled.cube.data.data[26, 0, 0], 0.74409089)
