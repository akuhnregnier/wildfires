#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from copy import deepcopy
from datetime import datetime

import cf_units
import iris
import iris.coord_categorisation
import numpy as np
import pytest
from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime
from joblib import Memory

import wildfires.data.datasets as wildfire_datasets
from test_datasets import data_availability
from wildfires.data.cube_aggregation import Datasets
from wildfires.data.datasets import get_centres

memory = Memory(location=os.environ.get("TMPDIR", "/tmp"))


# FIXME: Use Dataset.pretty and Dataset.pretty_variable_names attributes!!!


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
            var_name="var_name" + self.name,
            long_name="long_name" + self.name,
        )
        self.cubes = iris.cube.CubeList([cube])

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


DUMMY_DATASETS = [type(name, (DummyDataset,), {}) for name in ["A", "B", "C", "D"]]


@pytest.fixture(scope="function")
def big_dataset():
    big_dataset = DummyDataset()
    dummy_cube = deepcopy(big_dataset.cubes[0])
    dummy_cube.long_name = "second_name"
    dummy_cube.var_name = "second_var"
    big_dataset.cubes.append(dummy_cube)
    return big_dataset


@pytest.fixture(scope="module")
def sel():
    sel = Datasets()
    sel.add(DUMMY_DATASETS[0]())
    sel.add(DUMMY_DATASETS[1]())
    return sel


@pytest.fixture(scope="module")
def long_sel():
    long_sel = Datasets()
    for dataset in DUMMY_DATASETS:
        long_sel.add(dataset())
    return long_sel


def test_representations(sel):
    # Confirm expected output.
    all_all = sel.get(dataset_name="all", variable_format="all")
    assert all_all == {
        ("A", "A"): (("long_nameA", "long_nameA"),),
        ("B", "B"): (("long_nameB", "long_nameB"),),
    }

    # Confirm expected output.
    all_pretty = sel.get(dataset_name="all", variable_format="pretty")
    assert all_pretty == {"A": ("long_nameA",), "B": ("long_nameB",)}

    # Confirm expected output.
    all_raw = sel.get(dataset_name="all", variable_format="raw")
    assert all_raw == {"A": ("long_nameA",), "B": ("long_nameB",)}


def test_adding(sel):
    # Test guard against duplicated names.
    with pytest.raises(ValueError, match=re.escape("Match for datasets 'A' and 'A'.")):
        sel.add(DUMMY_DATASETS[0]())


def test_name_retrieval(sel):
    """Test that all names are retrieved correctly."""
    assert set(sel.raw_variable_names) == {"long_nameA", "long_nameB"}
    assert set(sel.pretty_variable_names) == {"long_nameA", "long_nameB"}


def test_equality(sel, long_sel):
    sel2 = Datasets().add(DUMMY_DATASETS[0]()).add(DUMMY_DATASETS[1]())

    assert sel2 == sel

    # TODO: Test equality while making use of pretty names.

    # See if different variable assignment orders affect equality.

    sel3 = Datasets().add(DUMMY_DATASETS[1]()).add(DUMMY_DATASETS[0]())

    assert sel == sel3


def test_removal(sel, long_sel):
    assert sel.remove_variables("long_nameA", inplace=False) == sel.select_variables(
        "long_nameB", inplace=False
    )

    assert set(
        long_sel.remove_variables(
            ("long_nameA", "long_nameC"), inplace=False
        ).raw_dataset_names
    ) == set(("B", "D"))

    assert long_sel.remove_datasets(
        ("D", "B"), inplace=False
    ) == long_sel.select_datasets(("A", "C"), inplace=False)


def test_creation(sel):
    comp_sel = Datasets((DUMMY_DATASETS[0](), DUMMY_DATASETS[1]()))
    assert comp_sel == sel


def test_addition(sel):
    test_sel = Datasets()
    orig_id = id(test_sel)
    test_sel += DUMMY_DATASETS[0]()
    test_sel += Datasets().add(DUMMY_DATASETS[1]())

    assert test_sel == sel
    assert id(test_sel) == orig_id

    with pytest.raises(ValueError, match=re.escape("Match for datasets 'A' and 'A'.")):
        _ = test_sel + sel

    test_sel2 = Datasets()
    orig_id2 = id(test_sel2)
    test_sel2 = test_sel2 + (DUMMY_DATASETS[0](), DUMMY_DATASETS[1]())

    assert test_sel2 == sel
    assert id(test_sel2) != orig_id2


@data_availability
def test_instances():
    hyde = wildfire_datasets.HYDE()
    sel1 = Datasets().add(hyde)
    sel2 = Datasets().add(hyde)
    assert sel1 == sel2

    sel3 = Datasets().add(wildfire_datasets.HYDE())
    assert sel1 == sel3


def test_pruning(big_dataset):
    assert (
        Datasets().add(DUMMY_DATASETS[0]()).remove_variables("long_nameA") == Datasets()
    )
    assert (
        Datasets()
        .add(DUMMY_DATASETS[0]())
        .add(DUMMY_DATASETS[1]())
        .remove_variables(("long_nameA", "long_nameB"))
        == Datasets()
    )

    assert Datasets().add(DUMMY_DATASETS[0]()).add(
        DUMMY_DATASETS[1]()
    ).remove_variables("long_nameA") == Datasets().add(DUMMY_DATASETS[1]())

    # big_dataset contains 2 cubes. Removing one of them should leave only 1.
    assert len(Datasets(big_dataset).remove_variables("second_name")[0]) == 1
