# -*- coding: utf-8 -*-
import iris
import numpy as np

from wildfires.data.cube_aggregation import Datasets
from wildfires.data.datasets import CCI_BurnedArea_MERIS_4_1


def test_MERIS_observed_area_mask():
    # Test retrieval of mask from full dataset.
    meris = CCI_BurnedArea_MERIS_4_1()
    mask = meris.get_observed_mask()
    ba = Datasets(meris).select_variables("CCI MERIS BA").cube

    assert isinstance(mask, iris.cube.Cube)
    assert mask.dtype in (np.bool, np.bool_)
    assert mask.shape == ba.shape

    # Test retrieval of mask from partial dataset.
    mask_partial = (
        Datasets(CCI_BurnedArea_MERIS_4_1())
        .remove_variables(CCI_BurnedArea_MERIS_4_1._observed_area["name"])
        .dataset.get_observed_mask()
    )
    assert np.all(mask.data == mask_partial.data)

    # This test relies on there being two ~15-day samples per month.
    mask_monthly = meris.get_observed_mask(frequency="monthly")

    target_shape = list(ba.shape)
    target_shape[0] /= 2

    assert isinstance(mask_monthly, iris.cube.Cube)
    assert mask_monthly.dtype in (np.bool, np.bool_)
    assert mask_monthly.shape == tuple(target_shape)
