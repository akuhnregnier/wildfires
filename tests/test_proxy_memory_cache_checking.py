# -*- coding: utf-8 -*-

from .utils import *  # noqa


def test_lazy_cache_checking(proxy_memory):
    """Proxy objects should not be realised when checking cache contents."""

    @proxy_memory.cache
    def f(x):
        return x + 1

    @proxy_memory.cache
    def f2(x):
        return x + 10

    # Generate cache entries.
    assert f(1) == 2
    assert f2(f(1)) == 12

    # Retrieve the proxy pointing to the cached result.
    f1_proxy = f(1)
    assert f2.check_in_store(f1_proxy)
    assert f2(f1_proxy) == 12

    assert not f1_proxy.__factory__.was_called


def test_chained_lazy_cache_checking(proxy_memory):
    """Proxy objects should not be realised when checking cache contents."""

    @proxy_memory.cache
    def input_proxy():
        return 1

    @proxy_memory.cache
    def f(x):
        return x + 1

    def f2(x, cache_check=False):
        if cache_check:
            return f.check_in_store(x)
        return f(x) + 10

    # Generate cache entries.
    assert f2(input_proxy()) == 12

    # Retrieve the proxy pointing to the cached result.
    in_proxy = input_proxy()
    assert f2(in_proxy, cache_check=True)
    assert f2(in_proxy) == 12

    assert not in_proxy.__factory__.was_called
