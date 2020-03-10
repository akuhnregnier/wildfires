# -*- coding: utf-8 -*-
from copy import deepcopy

import iris
import iris.coord_categorisation
import numpy as np
import pytest
from iris.time import PartialDateTime

import wildfires.data.datasets as wildfire_datasets
from wildfires.data.cube_aggregation import Datasets
from wildfires.data.datasets import dummy_lat_lon_cube

from .utils import data_availability

# FIXME: Use Dataset.pretty and Dataset.pretty_variable_names attributes!!!


class DummyDataset(wildfire_datasets.Dataset):
    def __init__(self, name=None):
        data = np.random.random((100, 100, 100))
        data = np.ma.MaskedArray(data, mask=data > 0.5)

        cube = dummy_lat_lon_cube(
            data,
            var_name="var_name" + self.name,
            long_name="long_name" + self.name if name is None else name,
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
    dummy_cube = deepcopy(big_dataset.cube)
    dummy_cube.long_name = "second_name"
    dummy_cube.var_name = "second_var"
    big_dataset.cubes.append(dummy_cube)
    return big_dataset


@pytest.fixture(scope="function")
def sel():
    return Datasets(dataset() for dataset in DUMMY_DATASETS[:2])


@pytest.fixture(scope="function")
def long_sel():
    return Datasets(dataset() for dataset in DUMMY_DATASETS)


def test_ordering():
    datasets = Datasets(
        (
            DUMMY_DATASETS[3](),
            DUMMY_DATASETS[0](),
            DUMMY_DATASETS[1](),
            DUMMY_DATASETS[2](),
        )
    )
    assert datasets.raw_dataset_names == ("A", "B", "C", "D")


def test_representations(sel):
    all_all = sel.state(dataset_name="all", variable_format="all")
    assert all_all == {
        ("A", "A"): (("long_nameA", "long_nameA"),),
        ("B", "B"): (("long_nameB", "long_nameB"),),
    }

    all_pretty = sel.state(dataset_name="all", variable_format="pretty")
    assert all_pretty == {"A": ("long_nameA",), "B": ("long_nameB",)}

    all_raw = sel.state(dataset_name="all", variable_format="raw")
    assert all_raw == {"A": ("long_nameA",), "B": ("long_nameB",)}


def test_adding(sel):
    # Test guard against duplicated names.
    with pytest.raises(ValueError, match="Match.*datasets.*'A.*and.*'A.*."):
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
    sel.show()
    long_sel.show()
    assert sel.remove_variables(
        "long_nameA", inplace=False, copy=True
    ) == sel.select_variables("long_nameB", inplace=False, copy=True)

    sel.show()
    long_sel.show()
    assert set(
        long_sel.remove_variables(
            ("long_nameA", "long_nameC"), inplace=False
        ).raw_dataset_names
    ) == set(("B", "D"))

    sel.show()
    long_sel.show()
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

    with pytest.raises(ValueError, match="Match.*datasets.*'A.*and.*'A.*."):
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

    orig_cube = sel3.cubes[0]
    comp_cube = sel3.select_variables(
        orig_cube.name(), inplace=False, strict=True
    ).cubes[0]

    assert id(orig_cube) == id(comp_cube)


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


def test_same_var_names():
    sel1 = Datasets(
        (
            type("A", (DummyDataset,), {})("var_name_1"),
            type("B", (DummyDataset,), {})("var_name_1"),
        )
    )
    sel2 = Datasets(
        (
            type("B", (DummyDataset,), {})("var_name_1"),
            type("A", (DummyDataset,), {})("var_name_1"),
        )
    )

    sel3 = sel1.select_variables("var_name_1", strict=False)

    assert sel1 == sel2 == sel3
