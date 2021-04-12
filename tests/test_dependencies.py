# -*- coding: utf-8 -*-

import pytest

from wildfires.cache import mark_dependency
from wildfires.exceptions import NotCachedError

from .utils import *  # noqa


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_dependencies(memory):
    """Test that when dependencies change, the cache gets invalidated."""

    @memory.cache
    @mark_dependency
    def f(x):
        return x + 1

    @memory.cache(dependencies=(f,))
    def f2(x):
        return f(x) + 10

    assert f(1) == 2
    assert f2(1) == 12

    # Defining the same functions as above should not invalidate the previous cache
    # entries.

    @memory.cache
    @mark_dependency
    def f(x):
        return x + 1

    @memory.cache(dependencies=(f,))
    def f2(x):
        return f(x) + 10

    assert f.check_in_store(1)
    assert f2.check_in_store(1)

    # However, redefining `f` should invalidate both cache entries.

    @memory.cache
    @mark_dependency
    def f(x):
        return x + 2

    @memory.cache(dependencies=(f,))
    def f2(x):
        return f(x) + 10

    with pytest.raises(NotCachedError):
        f.check_in_store(1)

    with pytest.raises(NotCachedError):
        f2.check_in_store(1)

    assert f(1) == 3
    assert f2(1) == 13


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_dependencies_2(memory):
    """Test that when dependencies change, the cache gets invalidated.

    This should work even if the cached function is marked as a possible dependency
    itself.

    """

    @memory.cache
    @mark_dependency
    def f(x):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f2(x):
        return f(x) + 10

    assert f(1) == 2
    assert f2(1) == 12

    # Defining the same functions as above should not invalidate the previous cache
    # entries.

    @memory.cache
    @mark_dependency
    def f(x):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f2(x):
        return f(x) + 10

    assert f.check_in_store(1)
    assert f2.check_in_store(1)

    # However, redefining `f` should invalidate both cache entries.

    @memory.cache
    @mark_dependency
    def f(x):
        return x + 2

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f2(x):
        return f(x) + 10

    with pytest.raises(NotCachedError):
        f.check_in_store(1)

    with pytest.raises(NotCachedError):
        f2.check_in_store(1)

    assert f(1) == 3
    assert f2(1) == 13


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_dependencies_default_args(memory):
    """Test cache is invalidated when default arguments of dependencies change."""

    @memory.cache
    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f2(x):
        return f(x) + 10

    assert f(1) == 2
    assert f2(1) == 12

    # Defining the same functions as above should not invalidate the previous cache
    # entries.

    @memory.cache
    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f2(x):
        return f(x) + 10

    assert f.check_in_store(1)
    assert f2.check_in_store(1)

    # However, redefining `f` should invalidate both cache entries.

    @memory.cache
    @mark_dependency
    def f(x=1):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f2(x):
        return f(x) + 10

    assert f.check_in_store(1)

    with pytest.raises(NotCachedError):
        f2.check_in_store(1)

    assert f(1) == 2
    assert f2(1) == 12


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_chained_dependencies(memory):
    """Test cache is invalidated when chained dependencies change."""

    @memory.cache
    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f1(x=0):
        return f(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    assert f(1) == 2
    assert f1(1) == -8
    assert f2(1) == 92

    # Defining the same functions as above should not invalidate the previous cache
    # entries.

    @memory.cache
    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f1(x=0):
        return f(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    for func in (f, f1, f2):
        assert func.check_in_store(1)

    # However, redefining `f` should invalidate all cache entries.

    @memory.cache
    @mark_dependency
    def f(x=0):
        return x + 2

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f1(x=0):
        return f(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    for func in (f, f1, f2):
        with pytest.raises(NotCachedError):
            func.check_in_store(1)

    assert f(1) == 3
    assert f1(1) == -7
    assert f2(1) == 93


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
@pytest.mark.parametrize("redefine", ["f", "f0", "f+f0"])
def test_multiple_chained_dependencies(memory, redefine):
    """Test cache is invalidated when chained dependencies change."""

    @memory.cache
    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache
    @mark_dependency
    def f0(x=0):
        return x + 3

    @memory.cache(dependencies=(f, f0))
    @mark_dependency
    def f1(x=0):
        return f(x) + f0(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    assert f(1) == 2
    assert f0(1) == 4
    assert f1(1) == -4
    assert f2(1) == 96

    # Defining the same functions as above should not invalidate the previous cache
    # entries.

    @memory.cache
    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache
    @mark_dependency
    def f0(x=0):
        return x + 3

    @memory.cache(dependencies=(f, f0))
    @mark_dependency
    def f1(x=0):
        return f(x) + f0(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    for func in (f, f0, f1, f2):
        assert func.check_in_store(1)

    # However, redefining either `f` or `f0` should invalidate all cache entries.

    if redefine == "f":

        @memory.cache
        @mark_dependency
        def f(x=0):
            return x + 2

    elif redefine == "f0":

        @memory.cache
        @mark_dependency
        def f0(x=0):
            return x + 2

    elif redefine == "f+f0":

        @memory.cache
        @mark_dependency
        def f(x=0):
            return x + 2

        @memory.cache
        @mark_dependency
        def f0(x=0):
            return x + 2

    else:
        raise ValueError("Unsupported 'redefine' value.")

    @memory.cache(dependencies=(f, f0))
    @mark_dependency
    def f1(x=0):
        return f(x) + f0(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    if redefine == "f":
        changed_funcs = (f, f1, f2)
    elif redefine == "f0":
        changed_funcs = (f0, f1, f2)
    elif redefine == "f+f0":
        changed_funcs = (f, f0, f1, f2)

    for func in changed_funcs:
        with pytest.raises(NotCachedError):
            func.check_in_store(1)

    if redefine == "f":
        assert f(1) == 3
        assert f0(1) == 4
        assert f1(1) == -3
        assert f2(1) == 97
    elif redefine == "f0":
        assert f(1) == 2
        assert f0(1) == 3
        assert f1(1) == -5
        assert f2(1) == 95
    elif redefine == "f+f0":
        assert f(1) == 3
        assert f0(1) == 3
        assert f1(1) == -4
        assert f2(1) == 96


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
def test_chained_uncached_dependencies(memory):
    """Test cache is invalidated when chained uncached dependencies change."""

    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f1(x=0):
        return f(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    assert f(1) == 2
    assert f1(1) == -8
    assert f2(1) == 92

    # Defining the same functions as above should not invalidate the previous cache
    # entries.

    @mark_dependency
    def f(x=0):
        return x + 1

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f1(x=0):
        return f(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    for func in (f1, f2):
        assert func.check_in_store(1)

    # However, redefining `f` should invalidate all cache entries.

    @mark_dependency
    def f(x=0):
        return x + 2

    @memory.cache(dependencies=(f,))
    @mark_dependency
    def f1(x=0):
        return f(x) - 10

    @memory.cache(dependencies=(f1,))
    @mark_dependency
    def f2(x):
        return f1(x) + 100

    for func in (f1, f2):
        with pytest.raises(NotCachedError):
            func.check_in_store(1)

    assert f(1) == 3
    assert f1(1) == -7
    assert f2(1) == 93
