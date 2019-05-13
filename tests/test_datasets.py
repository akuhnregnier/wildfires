#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pytest

from wildfires.data.datasets import DATA_DIR

data_availability = pytest.mark.skipif(
    not os.path.exists(DATA_DIR), reason="Data directory is unavailable."
)
