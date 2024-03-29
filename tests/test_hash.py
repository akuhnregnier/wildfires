# -*- coding: utf-8 -*-
import numpy as np
import pytest
from iris.time import PartialDateTime
from sklearn.ensemble import RandomForestRegressor

from .utils import *  # noqa


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_ndarray_get_hash(memory):
    get_hash = memory.get_hash

    data = np.arange(int(1e5))
    orig_hash = get_hash(data)
    data[int(1e4)] = 0
    assert get_hash(data) != orig_hash


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_maskedarray_get_hash(memory):
    get_hash = memory.get_hash

    data = np.ma.MaskedArray(np.arange(int(1e5)))
    data.mask = data.data > int(1e4)
    orig_hash = get_hash(data)
    data[int(1e4)] = 0
    assert get_hash(data) != orig_hash


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_dataset_get_hash(memory, dummy_dataset):
    get_hash = memory.get_hash

    # Test that the hash changes when data is changed.
    orig_hash = get_hash(dummy_dataset)
    dummy_dataset.cube.data.data[5, 180, 360] = 0
    mod_hash = get_hash(dummy_dataset)
    assert mod_hash != orig_hash

    # Test that the hash changes if the mask changes.
    dummy_dataset.cube.data.mask[5, 180, 360] = True
    mask_mod_hash = get_hash(dummy_dataset)
    assert mask_mod_hash != mod_hash

    # Test that the hash changes if a coordinate is altered.
    dummy_dataset.cube.coord("time").long_name = "testing"
    coord_mod_hash = get_hash(dummy_dataset)
    assert coord_mod_hash != mask_mod_hash


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_datasets_get_hash(memory, dummy_datasets):
    get_hash = memory.get_hash

    # Test that the hash changes when data is changed.
    orig_hash = get_hash(dummy_datasets)
    dummy_datasets.cube.data.data[5, 180, 360] = 0
    mod_hash = get_hash(dummy_datasets)
    assert mod_hash != orig_hash

    # Test that the hash changes if the mask changes.
    dummy_datasets.cube.data.mask[5, 180, 360] = True
    mask_mod_hash = get_hash(dummy_datasets)
    assert mask_mod_hash != mod_hash

    # Test that the hash changes if a coordinate is altered.
    dummy_datasets.cube.coord("time").long_name = "testing"
    coord_mod_hash = get_hash(dummy_datasets)
    assert coord_mod_hash != mask_mod_hash


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_estimator_get_hash(memory):
    get_hash = memory.get_hash

    # Generate training data.
    rng = np.random.default_rng(0)
    X = rng.random((10, 2))
    y = rng.random((10,))

    # Initialise the estimator.
    est = RandomForestRegressor(n_jobs=None)
    # Fit the estimator.
    est.fit(X, y)

    # Store the original hash of the model and its predict method.
    orig_est_hash = get_hash(est)
    orig_est_predict_hash = get_hash(est.predict)

    # Ensure 'n_jobs' has not been changed.
    assert est.n_jobs is None

    # Ensure this does not change as 'n_jobs' is changed.
    est.n_jobs = 1
    assert get_hash(est) == orig_est_hash
    assert get_hash(est.predict) == orig_est_predict_hash

    # Ensure 'n_jobs' has not been changed.
    assert est.n_jobs == 1

    # Ensure the hash changes as the model is retrained.

    # Generate new training data.
    rng = np.random.default_rng(1)
    X = rng.random((10, 2))
    y = rng.random((10,))
    # Fit the estimator anew.
    est.fit(X, y)
    # Ensure the hash has changed.
    assert get_hash(est) != orig_est_hash
    assert get_hash(est.predict) != orig_est_predict_hash


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_nested_ma_dict_get_hash(memory, dummy_datasets):
    get_hash = memory.get_hash

    nested_dict = {
        "a": np.ma.MaskedArray([1, 2, 3], mask=[1, 0, 1]),
        "b": {
            "c": np.ma.MaskedArray([10, 20, 30], mask=[0, 1, 1]),
            "d": [np.ma.MaskedArray([-1, -2, -3], mask=[1, 0, 0])],
        },
    }

    # Test that the hash changes when data is changed.
    orig_hash = get_hash(nested_dict)
    nested_dict["b"]["c"][1] = 100
    assert get_hash(nested_dict) != orig_hash


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_nested_iter_ma_dict_get_hash(memory, dummy_datasets):
    get_hash = memory.get_hash

    nested_dict = {
        "a": [np.ma.MaskedArray([1, 2, 3], mask=[1, 0, 1])],
        "b": {
            "c": [np.ma.MaskedArray([10, 20, 30], mask=[0, 1, 1])],
            "d": [np.ma.MaskedArray([-1, -2, -3], mask=[1, 0, 0])],
        },
    }

    # Test that the hash changes when data is changed.
    orig_hash = get_hash(nested_dict)
    nested_dict["b"]["c"][0][1] = 100
    assert get_hash(nested_dict) != orig_hash


@pytest.mark.parametrize("memory", ["iris", "cloudpickle", "proxy"], indirect=True)
def test_partialdatetime_get_hash(memory):
    get_hash = memory.get_hash

    dt = PartialDateTime(year=2000, month=1)
    orig_hash = get_hash(dt)
    dt.year = 2001
    assert get_hash(dt) != orig_hash
