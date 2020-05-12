# -*- coding: utf-8 -*-
import cf_units
import iris
import numpy as np
import pytest

from wildfires.data.datasets import MonthlyDataset, dummy_lat_lon_cube


class DummyDataset(MonthlyDataset):
    def __init__(self):
        data = np.random.random((10, 100, 100))
        data = np.ma.MaskedArray(data, mask=data > 0.5)

        self.cubes = iris.cube.CubeList(
            [
                dummy_lat_lon_cube(
                    data, units=cf_units.Unit("1"), long_name="A" + self.name
                )
            ]
        )


@pytest.mark.parametrize("inplace", (False, True))
def test_area_selection(inplace):
    """Test the sub-setting of a certain geographic area."""
    np.random.seed(1)

    region = {"latitude_range": (-10, 10), "longitude_range": (-10, 30)}

    dataset = DummyDataset()
    selected = dataset.select_data(**region, inplace=inplace)

    assert np.all(
        (selected.cube.coord("latitude").bounds[0][1] > region["latitude_range"][0])
        & (selected.cube.coord("latitude").bounds[-1][0] < region["latitude_range"][1])
    )
    assert np.all(
        (selected.cube.coord("longitude").bounds[0][1] > region["longitude_range"][0])
        & (
            selected.cube.coord("longitude").bounds[-1][0]
            < region["longitude_range"][1]
        )
    )

    if inplace:
        assert dataset is selected
        assert selected.cube.shape == dataset.cube.shape
    else:
        assert dataset is not selected
        assert selected.cube.shape[0] == dataset.cube.shape[0]
        assert np.all(
            np.array(selected.cube.shape[1:]) < np.array(dataset.cube.shape[1:])
        )
