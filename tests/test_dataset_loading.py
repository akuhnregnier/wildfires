# -*- coding: utf-8 -*-
import pytest

from wildfires.data.datasets import (
    CRU,
    CCI_BurnedArea_MERIS_4_1,
    CCI_BurnedArea_MODIS_5_1,
    Dataset,
    ERA5_CAPEPrecip,
    ERA5_DryDayPeriod,
    ERA5_TotalPrecipitation,
    GSMaP_dry_day_period,
    GSMaP_precipitation,
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
    else:
        inst = dataset()
        assert isinstance(inst, Dataset)
        assert inst.cubes

        if isinstance(inst, (CCI_BurnedArea_MERIS_4_1, CCI_BurnedArea_MODIS_5_1)):
            assert isinstance(inst.vegetation_class_names, list)
            assert len(inst.vegetation_class_names) == 18


@data_availability
@pytest.mark.parametrize("dataset", Dataset.datasets - slow_datasets)
def test_dataset_fast(dataset):
    dataset_test_func(dataset)


@pytest.mark.slow
@data_availability
@pytest.mark.parametrize("dataset", slow_datasets)
def test_dataset_slow(dataset):
    dataset_test_func(dataset)
