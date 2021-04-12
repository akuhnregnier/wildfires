# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from .utils import *  # noqa


@pytest.mark.parametrize("memory", ["cloudpickle", "proxy"], indirect=True)
@pytest.mark.parametrize(
    "value",
    [
        1,
        dummy_dataset,
        dummy_datasets,
        np.arange(10),
        np.ma.MaskedArray(np.arange(3), mask=[1, 0, 0]),
        pd.DataFrame(np.arange(1000)),
        pd.Series(np.arange(1000)),
    ],
)
def test_hash_roundtrip(memory, value):
    @memory.cache
    def f():
        return value

    # Calculate expected hash.
    expected_hash = memory.get_hash(value)

    # Generate cache entry.
    assert memory.get_hash(f()) == expected_hash

    # Then load the proxy and check the hash value.
    proxy = f()
    assert memory.get_hash(proxy) == expected_hash

    # Load the data for real and check again.
    str(proxy)
    assert memory.get_hash(proxy) == expected_hash
