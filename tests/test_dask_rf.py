# -*- coding: utf-8 -*-
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import pytest

from wildfires.dask_cx1.dask_rf import CachedResults


@pytest.fixture(params=[True, False])
def tempdir(request):
    """Run tests with enabled or disabled caching."""
    if request.param:
        temp = Path(mkdtemp()) / "cache_test"
        yield temp
        if temp.is_dir():
            rmtree(temp)
    else:
        yield None


def test_defaultdict(tempdir):
    """Test that the 2 nested defaultdicts work."""
    results = CachedResults(estimator_class=object, n_splits=10, cache_dir=tempdir)
    results["a"]["b"]["c"] = 10


def test_score_cache(tempdir):
    """Test caching of scores."""
    results = CachedResults(estimator_class=object, n_splits=10, cache_dir=tempdir)
    results["a"]["b"]["c"] = 123
    results.cache()
    if tempdir is None:
        assert (
            CachedResults(estimator_class=object, n_splits=10, cache_dir=tempdir) == {}
        )
    else:
        assert CachedResults(
            estimator_class=object, n_splits=10, cache_dir=tempdir
        ) == {"a": {"b": {"c": 123}}}


def test_estimator_cache(tempdir):
    """Test estimator caching."""
    results = CachedResults(estimator_class=object, n_splits=10, cache_dir=tempdir)
    with pytest.raises(KeyError):
        results.get_estimator("a")

    results.store_estimator("a", 123)

    if tempdir is None:
        with pytest.raises(KeyError):
            results.get_estimator("a")
    else:
        assert results.get_estimator("a") == 123


def test_results_copy(tempdir):
    results = CachedResults(estimator_class=object, n_splits=10, cache_dir=tempdir)
    results["a"]["b"]["c"] = 123
    copy = results.copy()
    assert copy == results
    assert copy.__dict__ == results.__dict__


def test_score_collation(tempdir):
    """Test collation of scores."""
    results = CachedResults(estimator_class=object, n_splits=10, cache_dir=tempdir)
    for i in range(4):
        results["parameters"]["test_score"][i] = i

    assert results.collate_scores()["parameters"]["test_score"] == list(range(4))
