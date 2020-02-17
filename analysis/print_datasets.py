#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Only purpose is to print a table of manually specified datasets for inclusion in a
paper or similar.

"""
import logging
import logging.config

from wildfires.analysis.analysis import print_dataset_times
from wildfires.analysis.plotting import (
    FigureSaver,
    cube_plotting,
    map_model_output,
    partial_dependence_plot,
)
from wildfires.data.cube_aggregation import (
    IGNORED_DATASETS,
    Datasets,
    get_all_datasets,
    prepare_selection,
)
from wildfires.data.datasets import (
    CHELSA,
    HYDE,
    MCD64CMQ_C6,
    VODCA,
    AvitabileThurnerAGB,
    CCI_BurnedArea_MERIS_4_1,
    CCI_BurnedArea_MODIS_5_1,
    Copernicus_SWI,
    ERA5_CAPEPrecip,
    ERA5_DryDayPeriod,
    ESA_CCI_Landcover_PFT,
    GFEDv4,
    GFEDv4s,
    GlobFluo_SIF,
    MOD15A2H_LAI_fPAR,
)
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # General setup.
    logging.config.dictConfig(LOGGING)

    selection = Datasets(
        (
            AvitabileThurnerAGB(),
            CCI_BurnedArea_MERIS_4_1(),
            CCI_BurnedArea_MODIS_5_1(),
            CHELSA(),
            Copernicus_SWI(),
            ERA5_CAPEPrecip(),
            ERA5_DryDayPeriod(),
            ESA_CCI_Landcover_PFT(),
            GFEDv4(),
            GFEDv4s(),
            GlobFluo_SIF(),
            HYDE(),
            MCD64CMQ_C6(),
            MOD15A2H_LAI_fPAR(),
            VODCA(),
        )
    )

    selection = selection.select_variables(
        [
            "CCI MERIS BA",
            "CCI MODIS BA",
            "GFED4 BA",
            "GFED4s BA",
            "MCD64CMQ BA",
            "AGBtree",
            "maximum temperature",
            "minimum temperature",
            "Soil Water Index with T=1",
            "Product of CAPE and Precipitation",
            "dry_day_period",
            "ShrubAll",
            "TreeAll",
            "pftCrop",
            "pftHerb",
            "SIF",
            "popd",
            "Fraction of Absorbed Photosynthetically Active Radiation",
            "Leaf Area Index",
            "Vegetation optical depth X-band (10.65 GHz - 10.7 GHz)",
        ]
    )

    print_dataset_times(selection, latex=True, lat_lon=True)
