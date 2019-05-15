#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pytest

import wildfires.data.datasets as wildfire_datasets
from wildfires.data.datasets import DATA_DIR


def data_is_available():
    """Check if DATA_DIR exists.

    Returns:
        bool

    """
    return os.path.exists(DATA_DIR)


data_availability = pytest.mark.skipif(
    not data_is_available(), reason="Data directory is unavailable."
)


@data_availability
def test_equality():
    hyde1 = wildfire_datasets.HYDE()
    hyde2 = wildfire_datasets.HYDE()

    assert hyde1 == hyde2
    assert hash(hyde1) == hash(hyde2)
    assert hyde1 is not hyde2

    latitudes = hyde1.cubes[0].coord("latitude")
    hyde1.cubes[0].coord("latitude").points = latitudes.points + 1

    assert hyde1 != hyde2
    assert hash(hyde1) != hash(hyde2)

    for dataset in (hyde1, hyde2):
        assert all(cube.has_lazy_data() for cube in dataset.cubes)
