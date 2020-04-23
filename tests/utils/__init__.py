# -*- coding: utf-8 -*-
import pytest
from psutil import virtual_memory

from wildfires.data.datasets import data_is_available


def available_memory_gb():
    return virtual_memory().total / 1024 ** 3


data_availability = pytest.mark.skipif(
    not data_is_available() or available_memory_gb() < 10,
    reason="Data directory is unavailable or there is not enough memory.",
)
