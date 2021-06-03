# -*- coding: utf-8 -*-
import pytest

from wildfires.utils import parallel_njit


@pytest.mark.parametrize("cache", [True, False])
def test_parallel_njit(cache):
    @parallel_njit(cache=cache)
    def func(x):
        return x + 1

    assert func(1) == 2
