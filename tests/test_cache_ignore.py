# -*- coding: utf-8 -*-


from .utils import *  # noqa


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_cache_ignore_args(memory):
    @memory.cache(ignore=["a", "b"])
    def f(x, a, b):
        return x

    assert f(1, 2, 3) == 1
    assert f.check_in_store(1, 2, 3)
    assert f.check_in_store(1, 3, 3)
    assert f.check_in_store(1, 2, 4)
    assert f.check_in_store(1, 3, 4)


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_cache_ignore_kwargs(memory):
    @memory.cache(ignore=["a", "b"])
    def f(x, a=None, b=1):
        return x

    assert f(1, 2, 3) == 1
    assert f.check_in_store(1, 2, b=3)
    assert f.check_in_store(1, a=2, b=3)
    assert f.check_in_store(1, b=3, a=2)
    assert f.check_in_store(b=3, a=2, x=1)
    assert f.check_in_store(1, 2, 3)
    assert f.check_in_store(1, 3, 3)
    assert f.check_in_store(1, 2, 4)
    assert f.check_in_store(1, 3, 4)


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_cache_ignore_args_kwargs(memory):
    @memory.cache(ignore=["a", "b"])
    def f(x, a, b=1):
        return x

    assert f(1, 2, 3) == 1
    assert f.check_in_store(1, 2, b=3)
    assert f.check_in_store(1, a=2, b=3)
    assert f.check_in_store(1, b=3, a=2)
    assert f.check_in_store(b=3, a=2, x=1)
    assert f.check_in_store(1, 2, 3)
    assert f.check_in_store(1, 3, 3)
    assert f.check_in_store(1, 2, 4)
    assert f.check_in_store(1, 3, 4)
