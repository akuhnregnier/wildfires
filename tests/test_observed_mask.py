# -*- coding: utf-8 -*-
import iris
import numpy as np
import pytest

from wildfires.data.cube_aggregation import Datasets
from wildfires.data.datasets import CCI_BurnedArea_MERIS_4_1
from test_datasets import data_availability


@data_availability
def test_MERIS_observed_area_mask():
    # Test retrieval of mask from full dataset.
    meris = CCI_BurnedArea_MERIS_4_1()
    mask = meris.get_observed_mask()
    ba = meris["CCI MERIS BA"]

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


@data_availability
@pytest.mark.parametrize("thres", [0.1, 0.8, 0.9])
def test_MERIS_obs_masked_dataset(thres):
    ndigits = 3
    ba_var = "CCI MERIS BA"

    # Reference masked burned area.
    meris = CCI_BurnedArea_MERIS_4_1()
    mask = meris.get_observed_mask(thres=thres)
    manual_ba = meris[ba_var]
    manual_ba.data.mask |= mask.data

    # Case to compare.
    masked_dataset = CCI_BurnedArea_MERIS_4_1.get_obs_masked_dataset(
        ba_var, thres=thres, ndigits=ndigits
    )
    ba = masked_dataset[ba_var]

    # Compare the masked data arrays. Where one of the two values being compared is
    # masked, the comparison result is also masked and does not influence the final
    # answer.
    assert np.all(np.isclose(ba.data, manual_ba.data))
    assert np.all(ba.data.mask == manual_ba.data.mask)

    rounded_thres = round(thres, ndigits)
    format_str = "_thres_{rounded_thres:0." + str(ndigits) + "f}"

    assert type(masked_dataset).__name__ == (
        "CCI_BurnedArea_MERIS_4_1_burned_area_"
        + format_str.format(rounded_thres=rounded_thres)
    )
