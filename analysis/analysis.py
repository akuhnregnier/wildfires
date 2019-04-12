#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import logging.config
import os
import pickle

from pprint import pprint

import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# import statsmodels.genmod.families.links as links

from wildfires.data.datasets import load_dataset_cubes, data_map_plot
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)


def combine_masks(data, invalid_values=(0,)):
    """Create a mask that shows where data is invalid.

    NOTE: Calls data.data, so lazy data is realised here!

    True - invalid data
    False - valid data

    Returns a boolean array with the same shape as the input data.

    """
    if hasattr(data, "mask"):
        mask = data.mask
        data_arr = data.data
    else:
        mask = np.zeros_like(data, dtype=bool)
        data_arr = data

    for invalid_value in invalid_values:
        mask |= np.isclose(data_arr, invalid_value)

    return mask


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)
    # TODO: Use iris cube long_name attribute to enter descriptive name
    # which will be used throughout data analysis and plotting (eg. for
    # selecting DataFrame columns).

    logger.info("Loading cubes")
    cubes = load_dataset_cubes()

    logger.info("Selecting cubes")

    selected_names = [
        "AGBtree",
        # 'mean temperature',
        # 'monthly precipitation',
        "maximum temperature",
        "minimum temperature",
        # 'Quality Flag with T=1',
        # 'Soil Water Index with T=60',
        # 'Soil Water Index with T=5',
        # 'Quality Flag with T=60',
        # 'Soil Water Index with T=20',
        # 'Soil Water Index with T=40',
        # 'Quality Flag with T=20',
        # 'Surface State Flag',
        "Soil Water Index with T=1",
        # 'Quality Flag with T=100',
        # 'Soil Water Index with T=15',
        # 'Quality Flag with T=10',
        # 'Soil Water Index with T=10',
        # 'Quality Flag with T=40',
        # 'Quality Flag with T=15',
        # 'Quality Flag with T=5',
        "Soil Water Index with T=100",
        "ShrubAll",
        "TreeAll",
        "pftBare",
        "pftCrop",
        "pftHerb",
        "pftNoLand",
        # 'pftShrubBD',
        # 'pftShrubBE',
        # 'pftShrubNE',
        # 'pftTreeBD',
        # 'pftTreeBE',
        # 'pftTreeND',
        # 'pftTreeNE',
        "monthly burned area",
        "dry_days",
        "precip",
        "SIF",
        # 'cropland',
        # 'ir_norice',
        # 'rf_norice',
        "popd",
        # 'conv_rangeland',
        # 'rangeland',
        # 'tot_rainfed',
        # 'pasture',
        # 'rurc',
        # 'rf_rice',
        # 'tot_rice',
        # 'uopp',
        # 'popc',
        # 'ir_rice',
        # 'urbc',
        # 'grazing',
        # 'tot_irri',
        "Combined Flash Rate Time Series",
        "VODorig",
        # 'Standard Deviation of LAI',
        "Fraction of Absorbed Photosynthetically Active Radiation",
        "Leaf Area Index",
        # 'Standard Deviation of fPAR',
        # 'Simard_Pinto_3DGlobalVeg_L3C',
        # 'biomass_totalag',
        # 'biomass_branches',
        # 'biomass_foliage',
        # 'biomass_roots',
        # 'biomass_stem'
    ]

    def selection_func(cube):
        return cube.name() in selected_names

    cubes = cubes.extract(iris.Constraint(cube_func=selection_func))

    pprint([cube.name() for cube in cubes])

    logger.info("Aggregating masks")
    # Use masking to extract only the relevant data.

    # Accumulate the masks for each dataset into a global mask.
    global_mask = np.zeros(cubes[0].shape, dtype=bool)
    for cube in cubes:
        # TODO: Find out the invalid values for the datasets that do not
        # have masks (if a mask hasn't been created before)!
        if hasattr(cube.data, "mask"):
            global_mask |= cube.data.mask
            global_mask |= np.isinf(cube.data.data)
            global_mask |= np.isnan(cube.data.data)
        else:
            global_mask |= np.isinf(cube.data)
            global_mask |= np.isnan(cube.data)

    logger.info("Using mask to select datasets")
    # Use this mask to select each dataset

    selected_datasets = []
    for cube in cubes:
        selected_data = cube.data[~global_mask]
        if hasattr(selected_data, "mask"):
            assert not np.any(selected_data.mask)
            selected_datasets.append((cube.name(), selected_data.data))
        else:
            selected_datasets.append((cube.name(), selected_data))

    dataset_names = [s[0] for s in selected_datasets]

    exog_name_map = {
        "diurnal temperature range": "diurnal temp range",
        "near-surface temperature minimum": "near-surface temp min",
        "near-surface temperature": "near-surface temp",
        "near-surface temperature maximum": "near-surface temp max",
        "wet day frequency": "wet day freq",
        "Volumetric Soil Moisture Monthly Mean": "soil moisture",
        "SIF": "SIF",
        "VODorig": "VOD",
        "Combined Flash Rate Monthly Climatology": "lightning rate",
        (
            "Population Density, v4.10 (2000, 2005, 2010, 2015, 2020)"
            ": 30 arc-minutes"
        ): "pop dens",
    }

    inclusion_names = {
        "near-surface temperature maximum",
        "Volumetric Soil Moisture Monthly Mean",
        "SIF",
        "VODorig",
        "diurnal temperature range",
        "wet day frequency",
        "Combined Flash Rate Monthly Climatology",
        "Population Density, v4.10 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes",
    }

    burned_area_name = "monthly burned area"

    exog_names = [
        exog_name_map.get(s[0], s[0])
        for s in selected_datasets
        if s[0] != burned_area_name
    ]
    raw_exog_data = np.hstack(
        [s[1].reshape(-1, 1) for s in selected_datasets if s[0] != burned_area_name]
    )

    endog_name = selected_datasets[dataset_names.index(burned_area_name)][0]
    endog_data = selected_datasets[dataset_names.index(burned_area_name)][1]

    # lim = int(5e3)
    lim = None
    endog_data = endog_data[:lim]
    raw_exog_data = raw_exog_data[:lim]

    endog_data = pd.Series(endog_data, name="burned area")
    exog_data = pd.DataFrame(raw_exog_data, columns=exog_names)

    exog_data["temperature range"] = (
        exog_data["maximum temperature"] - exog_data["minimum temperature"]
    )
    del exog_data["minimum temperature"]

    # Carry out log transformation for select variables.
    log_var_names = [
        "temperature range",
        # There are problems with negative surface
        # temperatures here!
        # 'near-surface temp max',
        "dry_days",
    ]

    for name in log_var_names:
        mod_data = exog_data[name] + 0.01
        assert np.all(mod_data >= (0.01 - 1e-8))
        exog_data["log " + name] = np.log(mod_data)
        del exog_data[name]

    # Carry out square root transformation
    sqrt_var_names = ["Combined Flash Rate Time Series", "popd"]
    for name in sqrt_var_names:
        assert np.all(exog_data[name] >= 0)
        exog_data["sqrt " + name] = np.sqrt(exog_data[name])
        del exog_data[name]

    """
    # Available links for Gaussian:
    [statsmodels.genmod.families.links.log,
     statsmodels.genmod.families.links.identity,
     statsmodels.genmod.families.links.inverse_power]

    """

    logger.info("Fitting model")

    model = sm.GLM(
        endog_data,
        exog_data,
        # family=sm.families.Gaussian(links.log)
        family=sm.families.Binomial(),
    )

    model_results = model.fit()
    print(model_results.summary())

    # fig = plt.figure(figsize=(9, 15))
    # sm.graphics.plot_partregress_grid(model_results, fig=fig)
    # plt.tight_layout()
    # plt.savefig('partregress.png')

    plt.figure(figsize=(12, 9))
    plt.hexbin(endog_data, model_results.fittedvalues, bins="log")
    plt.xlabel("real data")
    plt.ylabel("prediction")
    plt.savefig("real_vs_prediction.png")

    ba_predicted = np.zeros_like(global_mask, dtype=np.float64)
    ba_predicted[~global_mask] = model_results.fittedvalues
    ba_predicted = np.ma.MaskedArray(ba_predicted, mask=global_mask)
    data_map_plot(
        np.ma.mean(ba_predicted, axis=0),
        name="Predicted Mean Burned Area",
        filename="predicted_mean.png",
    )

    ba_data = np.zeros_like(global_mask, dtype=np.float64)
    ba_data[~global_mask] = endog_data.values
    ba_data = np.ma.MaskedArray(ba_data, mask=global_mask)
    data_map_plot(
        np.ma.mean(ba_data, axis=0),
        name="Mean observed burned area (GFEDv4)",
        filename="observed_mean_ba.png",
    )

    data_map_plot(
        np.mean(global_mask, axis=0),
        name="Mean available data mask",
        filename="mean_avail_mask.png",
    )
    data_map_plot(
        np.max(global_mask, axis=0),
        name="Min available data mask",
        filename="min_avail_mask.png",
    )
    data_map_plot(
        np.min(global_mask, axis=0),
        name="Max available data mask",
        filename="max_avail_mask.png",
    )
