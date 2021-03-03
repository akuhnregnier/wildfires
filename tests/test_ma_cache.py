# -*- coding: utf-8 -*-
import tempfile

import iris
import numpy as np
import pytest
from joblib import Memory

from wildfires.data import (
    Datasets,
    MonthlyDataset,
    dummy_lat_lon_cube,
    get_hash,
    ma_cache,
)


@pytest.fixture
def dummy_memory():
    tmp_dir = tempfile.TemporaryDirectory()
    yield Memory(tmp_dir.name)
    tmp_dir.cleanup()


class DummyDataset(MonthlyDataset):
    def __init__(self, shape=(10, 360, 720)):
        cube = dummy_lat_lon_cube(
            np.ma.MaskedArray(
                np.random.default_rng(0).normal(scale=100, size=shape),
                mask=np.zeros(shape, dtype=np.bool_),
            )
        )
        self.cubes = iris.cube.CubeList([cube])


def test_ndarray_get_hash():
    data = np.arange(int(1e5))
    orig_hash = get_hash(data)
    data[int(1e4)] = 0
    assert get_hash(data) != orig_hash


def test_maskedarray_get_hash():
    data = np.ma.MaskedArray(np.arange(int(1e5)))
    data.mask = data.data > int(1e4)
    orig_hash = get_hash(data)
    data[int(1e4)] = 0
    assert get_hash(data) != orig_hash


def test_dataset_get_hash():
    dataset = DummyDataset()

    # Test that the hash changes when data is changed.
    orig_hash = get_hash(dataset)
    dataset.cube.data.data[5, 180, 300] = 0
    mod_hash = get_hash(dataset)
    assert mod_hash != orig_hash

    # Test that the hash changes if the mask changes.
    dataset.cube.data.mask[5, 180, 300] = True
    mask_mod_hash = get_hash(dataset)
    assert mask_mod_hash != mod_hash

    # Test that the hash changes if a coordinate is altered.
    dataset.cube.coord("time").long_name = "testing"
    coord_mod_hash = get_hash(dataset)
    assert coord_mod_hash != mask_mod_hash


def test_datasets_get_hash():
    datasets = Datasets([DummyDataset()])

    # Test that the hash changes when data is changed.
    orig_hash = get_hash(datasets)
    datasets.cube.data.data[5, 180, 300] = 0
    mod_hash = get_hash(datasets)
    assert mod_hash != orig_hash

    # Test that the hash changes if the mask changes.
    datasets.cube.data.mask[5, 180, 300] = True
    mask_mod_hash = get_hash(datasets)
    assert mask_mod_hash != mod_hash

    # Test that the hash changes if a coordinate is altered.
    datasets.cube.coord("time").long_name = "testing"
    coord_mod_hash = get_hash(datasets)
    assert coord_mod_hash != mask_mod_hash


def test_ma_cache(dummy_memory):
    @ma_cache(memory=dummy_memory)
    def dummy_func(x, *args, **kwargs):
        return x

    dataset = DummyDataset()
    dataset.cube.data.mask[5, 180, 300] = True

    assert dataset == dummy_func(
        dataset,
        (1, 2, 3),
        2,
        3,
        a=10,
        b=20,
        c=np.ma.MaskedArray([1, 2, 3], mask=[1, 0, 1]),
    )

    # Test the loading too.
    loaded = dummy_func(
        dataset,
        (1, 2, 3),
        2,
        3,
        a=10,
        b=20,
        c=np.ma.MaskedArray([1, 2, 3], mask=[1, 0, 1]),
    )
    assert loaded is not dataset
    assert loaded.cube.data is not dataset.cube.data
    assert loaded == dataset
    assert get_hash(loaded.cube.data) == get_hash(dataset.cube.data)


def test_multiple_function_ma_cache(dummy_memory):
    """Ensure that the decorator can discern between different functions."""
    @ma_cache(memory=dummy_memory)
    def dummy_func1(x):
        return x + 1

    @ma_cache(memory=dummy_memory)
    def dummy_func2(x):
        return x + 2

    assert dummy_func1(0) == 1
    assert dummy_func2(0) == 2

    # Now test the previously cached versions.
    assert dummy_func1(0) == 1
    assert dummy_func2(0) == 2
