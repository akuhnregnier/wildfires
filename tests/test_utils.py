# -*- coding: utf-8 -*-
import numpy as np

from wildfires.utils import select_valid_subset


def test_subset():
    data = np.random.random((100, 100))
    data = np.ma.MaskedArray(data, mask=np.zeros_like(data, dtype=np.bool_))
    data.mask[:40] = True
    data.mask[:, :20] = True
    data.mask[:, -10:] = True

    assert np.all(np.isclose(select_valid_subset(data), data[40:, 20:90]))
    assert np.all(np.isclose(select_valid_subset(data, axis=0), data[40:]))
    assert np.all(np.isclose(select_valid_subset(data, axis=1), data[:, 20:90]))
    assert np.all(np.isclose(select_valid_subset(data, axis=(0, 1)), data[40:, 20:90]))
