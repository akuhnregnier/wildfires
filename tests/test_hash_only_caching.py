# -*- coding: utf-8 -*-
import pytest

from wildfires.cache.proxy_backend import HashProxy
from wildfires.exceptions import NotCachedError

from .utils import *  # noqa


def test_hash_only_caching(proxy_memory):
    @proxy_memory.cache(save_hashes_only=True)
    def f(x):
        """Function to cache."""
        return x + 1

    with pytest.raises(NotCachedError):
        f.check_in_store(0)

    # Generate the initial cache entry.
    initial_out = f(0)
    assert not isinstance(initial_out, HashProxy)
    assert initial_out == 1

    # Retrieve the hash value only.
    out = f(0)
    assert isinstance(out, HashProxy)

    cached_hash = out.hashed_value
    assert not out.__factory__.was_called

    # Call the factory function (which calls `f` now instead of loading from disk).
    assert out == 1

    assert proxy_memory.get_hash(out) == cached_hash
    assert out.__factory__.was_called
    assert isinstance(out, int)


def test_hash_only_caching_tuple(proxy_memory):
    @proxy_memory.cache(save_hashes_only=True)
    def f(x):
        """Function to cache."""
        return (x + 1, x + 2)

    with pytest.raises(NotCachedError):
        f.check_in_store(0)

    # Generate the initial cache entry.
    initial_out = f(0)
    assert not any(isinstance(v, HashProxy) for v in initial_out)
    assert initial_out == (1, 2)

    # Retrieve the hash value only.
    out = f(0)
    assert all(isinstance(v, HashProxy) for v in out)

    cached_hash_values = tuple(v.hashed_value for v in out)
    assert not any(v.__factory__.was_called for v in out)

    # Call the factory function (which calls `f` now instead of loading from disk).
    assert out == (1, 2)

    assert all(
        proxy_memory.get_hash(v) == cached_hash_values[i] for i, v in enumerate(out)
    )
    assert all(v.__factory__.was_called for v in out)
    assert all(isinstance(v, int) for v in out)
