# -*- coding: utf-8 -*-
from operator import attrgetter
from pathlib import Path

import pytest

from wildfires.data.datasets import (
    CHELSA,
    CRU,
    DATA_DIR,
    CCI_BurnedArea_MERIS_4_1,
    CCI_BurnedArea_MODIS_5_1,
    Dataset,
    ERA5_CAPEPrecip,
    ERA5_DryDayPeriod,
    ERA5_TotalPrecipitation,
    GSMaP_dry_day_period,
    GSMaP_precipitation,
    LIS_OTD_lightning_climatology,
)

from .utils import data_availability

slow_datasets = {
    CRU,
    ERA5_CAPEPrecip,
    ERA5_DryDayPeriod,
    ERA5_TotalPrecipitation,
    GSMaP_dry_day_period,
    GSMaP_precipitation,
}


def dataset_test_func(dataset):
    if getattr(dataset, "_not_implemented", False):
        with pytest.raises(NotImplementedError):
            inst = dataset()
    elif (dataset == CHELSA) and (
        not (Path(DATA_DIR) / "cache" / ("CHELSA" + ".nc")).is_file()
    ):
        pytest.skip("CHELSA dataset cache file not found.")
    elif (dataset == LIS_OTD_lightning_climatology) and (
        not list((Path(DATA_DIR) / "LIS_OTD_lightning_climatology").glob("*.nc"))
    ):
        pytest.skip("LIS_OTD_lightning_climatology .nc files not found.")
    elif (dataset == ERA5_TotalPrecipitation) and (
        not (Path(DATA_DIR) / "cache" / ("ERA5_TotalPrecipitation" + ".nc")).is_file()
    ):
        pytest.skip("ERA5_TotalPrecipitation dataset cache file not found.")
    elif (dataset == CRU) and (
        not (Path(DATA_DIR) / "cache" / ("CRU" + ".nc")).is_file()
    ):
        pytest.skip("CRU dataset cache file not found.")
    else:
        inst = dataset()
        assert isinstance(inst, Dataset)
        assert inst.cubes

        if isinstance(inst, (CCI_BurnedArea_MERIS_4_1, CCI_BurnedArea_MODIS_5_1)):
            assert isinstance(inst.vegetation_class_names, list)
            assert len(inst.vegetation_class_names) == 18


@data_availability
@pytest.mark.parametrize(
    "dataset", sorted(Dataset.datasets - slow_datasets, key=attrgetter("__name__"))
)
def test_dataset_fast(dataset):
    dataset_test_func(dataset)


@pytest.mark.slow
@data_availability
@pytest.mark.parametrize("dataset", sorted(slow_datasets, key=attrgetter("__name__")))
def test_dataset_slow(dataset):
    dataset_test_func(dataset)
