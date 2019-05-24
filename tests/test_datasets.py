#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from datetime import datetime

import cf_units
import iris
import numpy as np
import pytest
from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime

import wildfires.data.datasets as wildfire_datasets
from wildfires.data.datasets import DATA_DIR, get_centres


def data_is_available():
    """Check if DATA_DIR exists.

    Returns:
        bool

    """
    return os.path.exists(DATA_DIR)


data_availability = pytest.mark.skipif(
    not data_is_available(), reason="Data directory is unavailable."
)


class DummyDataset(wildfire_datasets.Dataset):
    def __init__(self):

        data = np.random.random((100, 100, 100))
        data = np.ma.MaskedArray(data, mask=data > 0.5)

        latitudes = iris.coords.DimCoord(
            get_centres(np.linspace(-90, 90, data.shape[1] + 1)),
            standard_name="latitude",
            units="degrees",
        )
        longitudes = iris.coords.DimCoord(
            get_centres(np.linspace(-180, 180, data.shape[2] + 1)),
            standard_name="longitude",
            units="degrees",
        )

        calendar = "gregorian"
        time_unit_str = "days since 1970-01-01 00:00:00"
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        datetimes = [datetime(2000, 1, 1)]
        while len(datetimes) < 100:
            datetimes.append(datetimes[-1] + relativedelta(months=+1))

        time_coord = iris.coords.DimCoord(
            cf_units.date2num(datetimes, time_unit_str, calendar),
            standard_name="time",
            units=time_unit,
        )
        coords = [(time_coord, 0), (latitudes, 1), (longitudes, 2)]
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=coords,
            units=cf_units.Unit("1"),
            long_name="A" + self.name,
        )
        self.cubes = iris.cube.CubeList([cube])

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


@pytest.fixture(scope="function")
def big_dataset():
    big_dataset = DummyDataset()
    dummy_cube = deepcopy(big_dataset.cubes[0])
    dummy_cube.long_name = "C"
    big_dataset.cubes.append(dummy_cube)
    dummy_cube = deepcopy(big_dataset.cubes[0])
    dummy_cube.long_name = "B"
    big_dataset.cubes.append(dummy_cube)
    return big_dataset


@data_availability
def test_equality():
    hyde1 = wildfire_datasets.HYDE()
    hyde2 = wildfire_datasets.HYDE()

    assert hyde1 == hyde2
    assert hyde1 is not hyde2

    latitudes = hyde1.cubes[0].coord("latitude")
    hyde1.cubes[0].coord("latitude").points = latitudes.points + 1

    assert hyde1 != hyde2

    for dataset in (hyde1, hyde2):
        assert all(cube.has_lazy_data() for cube in dataset.cubes)


@data_availability
def test_sorting(big_dataset):
    sorted_cube_names = tuple(sorted(cube.name() for cube in big_dataset))
    assert sorted_cube_names == tuple(cube.name() for cube in big_dataset)
