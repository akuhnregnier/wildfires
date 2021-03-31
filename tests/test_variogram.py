# -*- coding: utf-8 -*-
import numpy as np
import pytest

from wildfires.variogram import combine_multiple_stats, combine_stats, compute_variogram


def test_combine_stats():
    # Test combination of stats.
    X = np.random.default_rng(0).random(100)

    assert np.all(
        np.isclose(
            (100, np.mean(X), np.var(X)),
            combine_stats(
                30, 70, np.mean(X[:30]), np.mean(X[30:]), np.var(X[:30]), np.var(X[30:])
            ),
        )
    )

    assert np.all(
        np.isclose(
            (100, np.mean(X), np.var(X)),
            combine_multiple_stats(
                [30, 30, 40],
                [np.mean(X[:30]), np.mean(X[30:60]), np.mean(X[60:])],
                [np.var(X[:30]), np.var(X[30:60]), np.var(X[60:])],
            ),
        )
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_chunked_variogram(n_jobs):
    rng = np.random.default_rng(0)
    N = 3
    X = rng.random((N, N))
    coords = rng.random((X.size, 2))
    shared_kwargs = dict(bins=2, max_lag=150)
    assert np.all(
        np.isclose(
            compute_variogram(
                coords, X.ravel(), n_jobs=n_jobs, n_per_job=3, **shared_kwargs
            )[-1],
            compute_variogram(
                coords, X.ravel(), n_jobs=1, n_per_job=X.shape[0], **shared_kwargs
            )[-1],
        )
    )
