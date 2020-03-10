# -*- coding: utf-8 -*-
import pytest

from wildfires.data.datasets import data_is_available

data_availability = pytest.mark.skipif(
    not data_is_available(), reason="Data directory is unavailable."
)
