# -*- coding: utf-8 -*-
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import pytest

from wildfires.dask_cx1.dask_rf import CachedResults


@pytest.mark.parametrize(
    "estimator_class,n_splits,cache_dir",
    [(None, None, None), (object, 10, Path(mkdtemp()) / "cache_test")],
)
def test_defaultdict(estimator_class, n_splits, cache_dir):
    """Test that the 2 nested defaultdicts work."""
    results = CachedResults(
        estimator_class=estimator_class, n_splits=n_splits, cache_dir=cache_dir
    )
    results["a"]["b"]["c"] = 10
    if cache_dir is not None:
        rmtree(cache_dir)


@pytest.mark.parametrize(
    "estimator_class,n_splits,cache_dir",
    [(None, None, None), (object, 10, Path(mkdtemp()) / "cache_test")],
)
def test_score_cache(estimator_class, n_splits, cache_dir):
    """Test caching of scores."""
    results = CachedResults(
        estimator_class=estimator_class, n_splits=n_splits, cache_dir=cache_dir
    )
    results["a"]["b"]["c"] = 123
    results.cache()
    if cache_dir is None:
        assert (
            CachedResults(
                estimator_class=estimator_class, n_splits=n_splits, cache_dir=cache_dir
            )
            == {}
        )
    else:
        assert CachedResults(
            estimator_class=estimator_class, n_splits=n_splits, cache_dir=cache_dir
        ) == {"a": {"b": {"c": 123}}}
    if cache_dir is not None:
        rmtree(cache_dir)


@pytest.mark.parametrize(
    "estimator_class,n_splits,cache_dir",
    [(None, None, None), (object, 10, Path(mkdtemp()) / "cache_test")],
)
def test_estimator_cache(estimator_class, n_splits, cache_dir):
    """Test estimator caching."""
    results = CachedResults(
        estimator_class=estimator_class, n_splits=n_splits, cache_dir=cache_dir
    )
    with pytest.raises(KeyError):
        results.get_estimator("a")

    results.store_estimator("a", 123)

    if cache_dir is None:
        with pytest.raises(KeyError):
            results.get_estimator("a")
    else:
        assert results.get_estimator("a") == 123

    if cache_dir is not None:
        rmtree(cache_dir)


@pytest.mark.parametrize(
    "estimator_class,n_splits,cache_dir",
    [(None, None, None), (object, 10, Path(mkdtemp()) / "cache_test")],
)
def test_results_copy(estimator_class, n_splits, cache_dir):
    results = CachedResults(
        estimator_class=estimator_class, n_splits=n_splits, cache_dir=cache_dir
    )
    results["a"]["b"]["c"] = 123
    copy = results.copy()
    assert copy == results
    assert copy.__dict__ == results.__dict__

    if cache_dir is not None:
        rmtree(cache_dir)
