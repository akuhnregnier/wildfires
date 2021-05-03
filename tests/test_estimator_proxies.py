# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from wildfires.cache import get_proxied_estimator

from .utils import *  # noqa


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def X(rng):
    return rng.random((5, 2))


@pytest.fixture
def orig_estimator(rng, X):
    orig_estimator = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=10)
    y = rng.random((5, 1))
    orig_estimator.fit(X, y)
    return orig_estimator


def test_cached_predict(X, orig_estimator, proxy_memory):
    """Test that `estimator.predict` does not load the data."""
    get_hash = proxy_memory.get_hash
    estimator = get_proxied_estimator(orig_estimator, proxy_memory)

    assert not estimator.predict.__factory__._was_called
    assert not estimator.__factory__._was_called

    assert get_hash(estimator.predict) == get_hash(orig_estimator.predict)
    # This is the first time we are calculating the hash, so the factory should have
    # been called.
    assert estimator.predict.__factory__._was_called
    assert not estimator.__factory__._was_called

    # Get a fresh lazy instance of the estimator.
    estimator = get_proxied_estimator(orig_estimator, proxy_memory)

    # Get `predict` without realising the data.
    assert get_hash(estimator.predict) == get_hash(orig_estimator.predict)
    # This is the second time we are retrieving the hash, so the factory should not
    # have been called.
    assert not estimator.predict.__factory__._was_called
    assert not estimator.__factory__._was_called

    assert np.allclose(estimator.predict(X), orig_estimator.predict(X))
    assert not estimator.__factory__._was_called
    # Because we actually used the result, the factory should have been called.
    assert estimator.predict.__factory__._was_called
