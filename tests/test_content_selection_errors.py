# -*- coding: utf-8 -*-
import re
from operator import getitem

import iris
import numpy as np
import pytest

from wildfires.data import (
    DatasetNotFoundError,
    Datasets,
    MonthlyDataset,
    VariableNotFoundError,
    dummy_lat_lon_cube,
)
from wildfires.utils import strip_multiline


def get_dummy_dataset(name, pretty_name, variable_name, pretty_variable_name):
    def init_func(self):
        self.cubes = iris.cube.CubeList(
            [
                dummy_lat_lon_cube(
                    np.random.random((10, 10, 10)),
                    monthly=True,
                    long_name=variable_name,
                )
            ]
        )

    return type(
        name,
        (MonthlyDataset,),
        {
            "_pretty": pretty_name,
            "pretty_variable_names": {variable_name: pretty_variable_name},
            "__init__": init_func,
        },
    )


@pytest.fixture
def selection():
    return Datasets(get_dummy_dataset("Dummy", "Dummy Dataset", "a_var", "A Var")())


@pytest.mark.parametrize(
    "access_func,target",
    [
        (getitem, "Dummy Data"),
        (lambda x, y: x.select_datasets(y), "Dummy Data"),
        (getitem, get_dummy_dataset("Dummy2", "Dummy Dataset 2", "b_var", "B Var")()),
    ],
)
def test_dataset_selection_errors(selection, access_func, target):
    selection["Dummy"]
    selection["Dummy Dataset"]

    with pytest.raises(
        DatasetNotFoundError,
        match=re.escape(
            strip_multiline(
                f"""Dataset '{target}' could not be found.
            Available: raw names ('Dummy',)
            or pretty names ('Dummy Dataset',)."""
            )
        ),
    ):
        access_func(selection, target)


def test_variable_selection_errors(selection):
    selection.select_variables("a_var")
    selection.select_variables("A Var")

    with pytest.raises(
        VariableNotFoundError,
        match=re.escape(
            strip_multiline(
                """Variable 'a' could not be found.
            Available: raw names ('a_var',)
            or pretty names ('A Var',)."""
            )
        ),
    ):
        selection.select_variables("a")


def test_cube_selection(selection):
    assert isinstance(selection.dataset["a_var"], iris.cube.Cube)
    assert isinstance(selection.dataset["A Var"], iris.cube.Cube)

    with pytest.raises(
        VariableNotFoundError,
        match=re.escape(
            strip_multiline(
                f"""No cube could be found for index 'B'.
            Available: integer indices [0],
            raw names ('a_var',),
            or pretty names ('A Var',)."""
            )
        ),
    ):
        selection.dataset["B"]
