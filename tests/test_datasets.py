# -*- coding: utf-8 -*-
from copy import deepcopy

import cf_units
import iris
import numpy as np
import pytest
from iris.time import PartialDateTime

import wildfires.data.datasets as wildfire_datasets
from wildfires.data.cube_aggregation import Datasets
from wildfires.data.datasets import dummy_lat_lon_cube

from .utils import data_availability


class DummyDataset2(wildfire_datasets.Dataset):
    pretty = "Dummy2"
    pretty_variable_names = {"VarA": "A2", "VarB": "B2"}

    def __init__(self):
        data = np.random.random((100, 100, 100))
        data = np.ma.MaskedArray(data, mask=data > 0.5)

        self.cubes = iris.cube.CubeList(
            [
                dummy_lat_lon_cube(data, units=cf_units.Unit("1"), long_name="VarA"),
                dummy_lat_lon_cube(data, units=cf_units.Unit("1"), long_name="VarB"),
            ]
        )

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class DummyDataset(wildfire_datasets.Dataset):
    def __init__(self):
        data = np.random.random((100, 100, 100))
        data = np.ma.MaskedArray(data, mask=data > 0.5)

        self.cubes = iris.cube.CubeList(
            [
                dummy_lat_lon_cube(
                    data, units=cf_units.Unit("1"), long_name="A" + self.name
                )
            ]
        )

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


def test_pretty_names():
    datasets = Datasets(DummyDataset2())
    assert datasets.state("all", "all") == {
        ("DummyDataset2", "Dummy2"): (("VarA", "A2"), ("VarB", "B2"))
    }

    dummy3 = DummyDataset2()
    dummy3.pretty = "Dummy3"

    with pytest.raises(
        ValueError, match="Matching datasets.*DummyDataset2.*DummyDataset2.*"
    ):
        datasets.add(dummy3)

    assert datasets.select_datasets("DummyDataset2", inplace=False) == datasets
    assert datasets.select_datasets("Dummy2", inplace=False) == datasets
    assert datasets.select_variables(("B2", "A2"), inplace=False) == datasets
    assert datasets.select_variables(("VarB", "A2"), inplace=False) == datasets
    assert datasets.select_variables(("B2", "VarA"), inplace=False) == datasets


def test_duplication():
    class DummyDataset(wildfire_datasets.Dataset):
        pretty = "Dummy"
        pretty_variable_names = {"VarA": "A2", "VarB": "B2", "VarC": "B2"}

        def __init__(self):
            data = np.random.random((100, 100, 100))
            data = np.ma.MaskedArray(data, mask=data > 0.5)

            self.cubes = iris.cube.CubeList(
                [
                    dummy_lat_lon_cube(
                        data, units=cf_units.Unit("1"), long_name="VarA"
                    ),
                    dummy_lat_lon_cube(
                        data, units=cf_units.Unit("1"), long_name="VarB"
                    ),
                    dummy_lat_lon_cube(
                        data, units=cf_units.Unit("1"), long_name="VarC"
                    ),
                ]
            )

        def get_monthly_data(
            self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
        ):
            return self.select_monthly_from_monthly(start, end)

    # This error is only raised when the `cubes` property is accessed.
    with pytest.raises(AssertionError, match="All variable names should be unique."):
        DummyDataset().cubes
