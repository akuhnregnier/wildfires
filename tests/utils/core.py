# -*- coding: utf-8 -*-
import tempfile

import iris
import numpy as np
import pytest

from wildfires.cache import CloudpickleMemory, IrisMemory, ProxyMemory
from wildfires.data import Datasets, MonthlyDataset, dummy_lat_lon_cube


@pytest.fixture
def tmp_dir():
    tmp_dir = tempfile.TemporaryDirectory()
    yield tmp_dir.name
    tmp_dir.cleanup()


@pytest.fixture
def proxy_memory(tmp_dir):
    return ProxyMemory(tmp_dir)


@pytest.fixture
def iris_memory(tmp_dir):
    return IrisMemory(tmp_dir)


@pytest.fixture
def memory(request, tmp_dir):
    if request.param == "iris":
        return IrisMemory(tmp_dir)
    elif request.param == "cloudpickle":
        return CloudpickleMemory(tmp_dir)
    elif request.param == "proxy":
        return ProxyMemory(tmp_dir)


@pytest.fixture
def dummy_dataset():
    class DummyDataset(MonthlyDataset):
        def __init__(self):
            self.cubes = iris.cube.CubeList(
                [
                    dummy_lat_lon_cube(
                        np.ma.MaskedArray(
                            np.random.default_rng(0).random((10, 360, 720)),
                            mask=np.zeros((10, 360, 720), dtype=np.bool_),
                        )
                    )
                ]
            )

    return DummyDataset()


@pytest.fixture
def dummy_datasets(dummy_dataset):
    return Datasets([dummy_dataset])
