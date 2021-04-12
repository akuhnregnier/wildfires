# -*- coding: utf-8 -*-


from wildfires.cache.proxy_backend import HashProxy

from .utils import *  # noqa


def test_cache_checking(proxy_memory):
    @proxy_memory.cache
    def f(x):
        """Function to be cached."""
        return (x + 1, x + 2)

    # Call the function to generate the cache entry.
    assert f(1) == (2, 3)

    # Retrieve the proxies.
    proxies = f(1)

    assert isinstance(proxies, tuple)
    assert all(isinstance(p, HashProxy) for p in proxies)
