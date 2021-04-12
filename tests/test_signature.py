# -*- coding: utf-8 -*-

import pytest

from .utils import *  # noqa


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_same_signature(memory):
    """The way a cached function is called should not affect the caching behaviour."""

    @memory.cache
    def f(x, y=10):
        return x + y

    assert f(1) == 11
    assert f.check_in_store(1)
    assert f.check_in_store(1, y=10)
    assert f.check_in_store(1, 10)
    assert f.check_in_store(x=1, y=10)
    assert f.check_in_store(y=10, x=1)


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_same_signature_kwargs(memory):
    """The way a cached function is called should not affect the caching behaviour."""

    @memory.cache
    def f(x, y=10, **kwargs):
        return x + y + sum(map(ord, kwargs.keys())) + sum(kwargs.values())

    assert f(1, a=10) == 118
    assert f.check_in_store(1, y=10, a=10)
    assert f.check_in_store(1, a=10, y=10)
    assert f.check_in_store(1, 10, a=10)


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_same_signature_args_kwargs(memory):
    """The way a cached function is called should not affect the caching behaviour."""

    @memory.cache
    def f(x, *my_args, y=4, **my_kwargs):
        return (
            x
            + sum(my_args)
            + y
            + sum(map(ord, my_kwargs.keys()))
            + sum(my_kwargs.values())
        )

    assert f(1, 2, 3, y=4, a=10, b=20) == 235
    assert f.check_in_store(1, 2, 3, a=10, b=20)
