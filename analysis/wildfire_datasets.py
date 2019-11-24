#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting of regional, climatological burned area timeseries.

These timeseries are plotted for 5 different burned area datasets.

"""
import warnings
from copy import deepcopy
from functools import reduce
from itertools import islice
from pprint import pprint

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from wildfires.analysis.plotting import cube_plotting
from wildfires.data.cube_aggregation import Datasets, prepare_selection
from wildfires.data.datasets import (
    MCD64CMQ_C6,
    CCI_BurnedArea_MERIS_4_1,
    CCI_BurnedArea_MODIS_5_1,
    GFEDv4,
    GFEDv4s,
    regions_GFED,
)
from wildfires.utils import land_mask as get_land_mask
from wildfires.utils import match_shape

if __name__ == "__main__":
    fire_datasets = Datasets(
        (
            fire_dataset()
            for fire_dataset in (
                GFEDv4s,
                GFEDv4,
                CCI_BurnedArea_MODIS_5_1,
                MCD64CMQ_C6,
                CCI_BurnedArea_MERIS_4_1,
            )
        )
    ).select_variables(
        ["CCI MERIS BA", "CCI MODIS BA", "GFED4 BA", "GFED4s BA", "MCD64CMQ BA"]
    )

    monthly, mean, climatology = prepare_selection(fire_datasets, which="all")

    pprint(list(monthly))

    land_mask = ~get_land_mask()

    no_fire_mask = np.all(
        reduce(np.logical_and, (np.isclose(cube.data, 0) for cube in monthly.cubes)),
        axis=0,
    )

    for fire_datasets in (monthly, mean, climatology):
        fire_datasets.homogenise_masks()
        for cube in fire_datasets.cubes:
            cube.data.mask |= reduce(
                np.logical_or,
                (match_shape(mask, cube.shape) for mask in (land_mask, no_fire_mask)),
            )

    mpl.rc("figure", figsize=(14, 6))
    for cube, name in zip(mean.cubes, mean.pretty_variable_names):
        m = cube.collapsed(
            ("latitude", "longitude"),
            iris.analysis.MEAN,
            weights=iris.analysis.cartography.area_weights(cube),
        ).data
        cube_plotting(cube, log=True, title=name + f" {m:0.5f}")

    # Seasonality

    regions = regions_GFED()
    # Skip region index 0, ie. the ocean.
    for region_index in islice(regions.attributes["regions"], 1, None):
        region_name = regions.attributes["regions"][region_index]
        region_mask = regions.data != region_index
        fig = plt.figure(figsize=(20, 8))
        axes = (plt.subplot(1, 2, 1), plt.subplot(1, 2, 2, projection=ccrs.Robinson()))
        for cube, name in zip(
            deepcopy(climatology.cubes), climatology.pretty_variable_names
        ):
            cube.data.mask |= match_shape(region_mask, cube.shape)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*DEFAULT.*")
                axes[0].plot(
                    range(1, 13),
                    cube.collapsed(
                        ("latitude", "longitude"),
                        iris.analysis.MEAN,
                        weights=iris.analysis.cartography.area_weights(cube),
                    ).data,
                    label=name,
                )
        axes[0].legend(loc="best")
        axes[0].set_ylabel("Average Burned Area Fraction")
        axes[0].set_xlabel("Month")
        axes[0].set_yscale("log")
        vis_cube = deepcopy(mean.cubes[0])
        vis_cube.data.mask |= region_mask
        if np.all(vis_cube.data.mask):
            print(f"No data for {region_name}")
        else:
            cube_plotting(vis_cube, ax=axes[1], log=True, title=region_name)
