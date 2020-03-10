#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from wildfires.data.datasets import (
    CHELSA,
    CRU,
    HYDE,
    MCD64CMQ_C6,
    VODCA,
    AvitabileAGB,
    AvitabileThurnerAGB,
    CarvalhaisGPP,
    CCI_BurnedArea_MERIS_4_1,
    CCI_BurnedArea_MODIS_5_1,
    Copernicus_SWI,
    Dataset,
    ERA5_CAPEPrecip,
    ERA5_DryDayPeriod,
    ERA5_TotalPrecipitation,
    ESA_CCI_Fire,
    ESA_CCI_Landcover,
    ESA_CCI_Landcover_PFT,
    ESA_CCI_Soilmoisture,
    ESA_CCI_Soilmoisture_Daily,
    GFEDv4,
    GFEDv4s,
    GlobFluo_SIF,
    GPW_v4_pop_dens,
    GSMaP_dry_day_period,
    GSMaP_precipitation,
    LIS_OTD_lightning_climatology,
    LIS_OTD_lightning_time_series,
    Liu_VOD,
    MOD15A2H_LAI_fPAR,
    Simard_canopyheight,
    Thurner_AGB,
)

from .utils import data_availability


@data_availability
def test_AvitabileAGB():
    try:
        inst = AvitabileAGB()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_AvitabileThurnerAGB():
    try:
        inst = AvitabileThurnerAGB()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_CCI_BurnedArea_MERIS_4_1():
    try:
        inst = CCI_BurnedArea_MERIS_4_1()
        assert isinstance(inst.vegetation_class_names, list)
        assert len(inst.vegetation_class_names) == 18
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_CCI_BurnedArea_MODIS_5_1():
    try:
        inst = CCI_BurnedArea_MODIS_5_1()
        assert isinstance(inst.vegetation_class_names, list)
        assert len(inst.vegetation_class_names) == 18
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_CHELSA():
    try:
        inst = CHELSA()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@pytest.mark.slow
@data_availability
def test_CRU():
    try:
        inst = CRU()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_CarvalhaisGPP():
    try:
        inst = CarvalhaisGPP()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_Copernicus_SWI():
    try:
        inst = Copernicus_SWI()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@pytest.mark.slow
@data_availability
def test_ERA5_CAPEPrecip():
    try:
        inst = ERA5_CAPEPrecip()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@pytest.mark.slow
@data_availability
def test_ERA5_DryDayPeriod():
    try:
        inst = ERA5_DryDayPeriod()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@pytest.mark.slow
@data_availability
def test_ERA5_TotalPrecipitation():
    try:
        inst = ERA5_TotalPrecipitation()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_ESA_CCI_Fire():
    try:
        inst = ESA_CCI_Fire()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_ESA_CCI_Landcover():
    try:
        inst = ESA_CCI_Landcover()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_ESA_CCI_Landcover_PFT():
    try:
        inst = ESA_CCI_Landcover_PFT()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_ESA_CCI_Soilmoisture():
    try:
        inst = ESA_CCI_Soilmoisture()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_ESA_CCI_Soilmoisture_Daily():
    try:
        inst = ESA_CCI_Soilmoisture_Daily()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_GFEDv4():
    try:
        inst = GFEDv4()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_GFEDv4s():
    try:
        inst = GFEDv4s()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_GPW_v4_pop_dens():
    try:
        inst = GPW_v4_pop_dens()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@pytest.mark.slow
@data_availability
def test_GSMaP_dry_day_period():
    try:
        inst = GSMaP_dry_day_period()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@pytest.mark.slow
@data_availability
def test_GSMaP_precipitation():
    try:
        inst = GSMaP_precipitation()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_GlobFluo_SIF():
    try:
        inst = GlobFluo_SIF()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_HYDE():
    try:
        inst = HYDE()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_LIS_OTD_lightning_climatology():
    try:
        inst = LIS_OTD_lightning_climatology()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_LIS_OTD_lightning_time_series():
    try:
        inst = LIS_OTD_lightning_time_series()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_Liu_VOD():
    try:
        inst = Liu_VOD()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_MCD64CMQ_C6():
    try:
        inst = MCD64CMQ_C6()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_MOD15A2H_LAI_fPAR():
    try:
        inst = MOD15A2H_LAI_fPAR()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_Simard_canopyheight():
    try:
        inst = Simard_canopyheight()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_Thurner_AGB():
    try:
        inst = Thurner_AGB()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes


@data_availability
def test_VODCA():
    try:
        inst = VODCA()
    except NotImplementedError:
        return
    assert isinstance(inst, Dataset)
    assert inst.cubes
