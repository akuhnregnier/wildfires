#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data analysis."""
import logging
import logging.config
import os
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import wildfires.utils as utils
from wildfires.analysis.plotting import (
    cube_plotting,
    map_model_output,
    partial_dependence_plot,
)
from wildfires.analysis.processing import log_map, map_name, vif
from wildfires.data.cube_aggregation import get_all_datasets, prepare_selection
from wildfires.data.datasets import dataset_times
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)
logging.config.dictConfig(LOGGING)


def log_mapping(key):
    if "log" in key:
        return False
    if key.lower() in {"monthly burned area", "popd"}:
        return True
    if " ".join(key.lower().split(" ")[1:]) in {"monthly burned area", "popd"}:
        return True
    return False


if __name__ == "__main__":
    normal_size = 9.0
    normal_coast_linewidth = 0.5
    dpi = 600

    logging.config.dictConfig(LOGGING)
    selection = get_all_datasets(
        ignore_names=(
            "AvitabileAGB",
            "CRU",
            "ESA_CCI_Fire",
            "ESA_CCI_Landcover",
            "ESA_CCI_Soilmoisture",
            "ESA_CCI_Soilmoisture_Daily",
            "GFEDv4s",
            "GPW_v4_pop_dens",
            "LIS_OTD_lightning_time_series",
            "Simard_canopyheight",
            "Thurner_AGB",
        )
    )
    selected_names = [
        "AGBtree",
        "maximum temperature",
        "minimum temperature",
        "Soil Water Index with T=1",
        "ShrubAll",
        "TreeAll",
        "pftBare",
        "pftCrop",
        "pftHerb",
        "monthly burned area",
        "dry_days",
        "dry_day_period",
        "precip",
        "SIF",
        "popd",
        "Combined Flash Rate Monthly Climatology",
        "VODorig",
        "Fraction of Absorbed Photosynthetically Active Radiation",
        "Leaf Area Index",
    ]

    selection = selection.select_variables(selected_names, strict=True)
    selection.show("pretty")

    monthly_datasets, mean_datasets, climatology_datasets = prepare_selection(selection)

    min_time, max_time, times_df = dataset_times(selection.datasets)
    # print(times_df)
    print(times_df.to_latex(index=False))

    mpl.rcParams["figure.figsize"] = (8, 5)

    # Get land mask.
    land_mask = ~utils.land_mask(n_lon=1440)

    fig = cube_plotting(land_mask.astype("int64"), title="Land Mask", cmap="Reds_r")

    filename = os.path.expanduser(os.path.join("~/tmp/to_send", "land_mask" + ".pdf"))
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    # Define a latitude mask which ignores data beyond 60 degrees, as the precipitation data does not extend to those latitudes.
    lats = mean_datasets.cubes[0].coord("latitude").points
    lons = mean_datasets.cubes[0].coord("longitude").points

    latitude_grid = np.meshgrid(lats, lons, indexing="ij")[0]
    lat_mask = np.abs(latitude_grid) > 60

    # Make a deep copy so that the original cubes are preserved.
    lat_land_datasets = mean_datasets.copy(deep=True)

    for cube in lat_land_datasets.cubes:
        cube.data.mask[lat_mask] = True
        cube.data.mask[land_mask] = True

    n_cols = 4
    n_plots = len(lat_land_datasets.cubes)

    mpl.rcParams["figure.figsize"] = (20, 12)

    fig, axes = plt.subplots(
        nrows=int(np.ceil(float(n_plots) / n_cols)), ncols=n_cols, squeeze=False
    )
    axes = axes.flatten()
    for (i, (ax, feature)) in enumerate(zip(axes, range(n_plots))):
        ax.hist(
            lat_land_datasets.cubes[feature].data.data[
                ~lat_land_datasets.cubes[feature].data.mask
            ],
            density=True,
            bins=70,
        )
        ax.set_xlabel(lat_land_datasets.pretty_variable_names[feature])
        ax.set_yscale("log")

    for ax in axes[n_plots:]:
        ax.set_axis_off()

    plt.tight_layout()

    # Get names of all the cubes we have access to.
    from pprint import pprint

    pprint(lat_land_datasets.raw_variable_names)

    figsize = (5, 3.8)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = normal_size

    cube = lat_land_datasets.select_variables(
        "monthly burned area", inplace=False
    ).cubes[0]
    fig = cube_plotting(
        cube,
        cmap="Reds",
        log=True,
        label="ln(Fraction)",
        title="Log Mean Burned Area (GFEDv4)",
        coastline_kwargs={"linewidth": normal_coast_linewidth},
    )
    filename = os.path.expanduser(
        os.path.join("~/tmp/to_send", cube.name().replace(" ", "_") + ".pdf")
    )
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    figsize = (4, 2.7)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = normal_size

    cube = lat_land_datasets.select_variables("AGBtree", inplace=False).cubes[0]
    fig = cube_plotting(
        cube,
        cmap="viridis",
        log=True,
        label=r"kg m$^{-2}$",
        title="AGBtree",
        coastline_kwargs={"linewidth": normal_coast_linewidth},
    )
    filename = os.path.expanduser(
        os.path.join("~/tmp/to_send", cube.name().replace(" ", "_") + ".pdf")
    )
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    sif_cube = lat_land_datasets.select_variables("SIF", inplace=False).cubes[0]
    sif_cube.data.mask |= sif_cube.data.data > 20
    sif_cube.data.mask |= sif_cube.data.data < 0

    lightning_cube = lat_land_datasets.select_variables(
        "Combined Flash Rate Monthly Climatology", inplace=False
    ).cubes[0]
    lightning_cube.data.mask |= lightning_cube.data.data < 0

    filled_datasets = lat_land_datasets.copy(deep=True).fill(
        land_mask=land_mask, lat_mask=lat_mask
    )

    figsize = (4, 2.7)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = normal_size

    cube = filled_datasets.select_variables("AGBtree", inplace=False).cubes[0]
    fig = cube_plotting(
        cube,
        cmap="viridis",
        log=True,
        label=r"kg m$^{-2}$",
        title="AGBtree",
        coastline_kwargs={"linewidth": normal_coast_linewidth},
    )

    mpl.rcParams["figure.figsize"] = (8, 5)
    fig = cube_plotting(
        filled_datasets.select_variables(
            "monthly burned area", strict=True, inplace=False
        )
        .cubes[0]
        .data.mask.astype("int64"),
        title="Land Mask",
        cmap="Reds_r",
    )

    filename = os.path.expanduser(os.path.join("~/tmp/to_send", "land_mask2" + ".pdf"))
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    figsize = (4, 2.7)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = normal_size

    cube = filled_datasets.select_variables(
        "AGBtree", strict=True, inplace=False
    ).cubes[0]
    fig = cube_plotting(
        cube,
        cmap="viridis",
        log=True,
        label=r"kg m$^{-2}$",
        title="AGBtree (interpolated)",
        coastline_kwargs={"linewidth": normal_coast_linewidth},
    )
    filename = os.path.expanduser(
        os.path.join(
            "~/tmp/to_send", cube.name().replace(" ", "_") + "_interpolated" + ".pdf"
        )
    )
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    burned_area_cube = filled_datasets.select_variables(
        "monthly burned area", strict=True, inplace=False
    ).cubes[0]
    endog_data = pd.Series(burned_area_cube.data.data[~burned_area_cube.data.mask])
    names = []
    data = []
    for cube in filled_datasets.cubes:
        if cube.name() != "monthly burned area":
            names.append(cube.name())
            data.append(cube.data.data[~cube.data.mask].reshape(-1, 1))

    exog_data = pd.DataFrame(np.hstack(data), columns=names)

    exog_data["temperature range"] = (
        exog_data["maximum temperature"] - exog_data["minimum temperature"]
    )
    del exog_data["minimum temperature"]

    print("Names before:")
    print(exog_data.columns)

    # Carry out log transformation for select variables.
    log_var_names = ["temperature range", "dry_days", "dry_day_period"]

    for name in log_var_names:
        mod_data = exog_data[name] + 0.01
        assert np.all(mod_data >= (0.01 - 1e-8)), "{:}".format(name)
        exog_data["log " + name] = np.log(mod_data)
        del exog_data[name]

    # Carry out square root transformation
    sqrt_var_names = ["Combined Flash Rate Monthly Climatology", "popd"]
    for name in sqrt_var_names:
        assert np.all(exog_data[name] >= 0), "{:}".format(name)
        exog_data["sqrt " + name] = np.sqrt(exog_data[name])
        del exog_data[name]

    print("Names after:")
    print(exog_data.columns)

    vifs = vif(exog_data)
    print(vifs.to_string(index=False, float_format="{:0.1f}".format))

    thres = 6
    vifs.loc[
        vifs["Name"] == "Fraction of Absorbed Photosynthetically Active Radiation",
        "Name",
    ] = "FAPAR"
    print(
        vifs.loc[vifs["VIF"] < thres].to_latex(
            index=False, float_format="{:0.1f}".format
        )
    )
    print(
        vifs.loc[vifs["VIF"] >= thres].to_latex(
            index=False, float_format="{:0.1f}".format
        )
    )

    # Remove redundant variables.
    exog_data2 = deepcopy(exog_data)
    for key in (
        "Fraction of Absorbed Photosynthetically Active Radiation",
        "Leaf Area Index",
        "precip",
        "pftBare",
        "TreeAll",
        "log dry_days",
    ):
        del exog_data2[key]

    vifs2 = vif(exog_data2)
    print(vifs2.to_string(index=False, float_format="{:0.1f}".format))

    thres = 6
    print(
        vifs2.loc[vifs2["VIF"] < thres].to_latex(
            index=False, float_format="{:0.1f}".format
        )
    )
    print(
        vifs2.loc[vifs2["VIF"] >= thres].to_latex(
            index=False, float_format="{:0.1f}".format
        )
    )

    model = sm.GLM(endog_data, exog_data2, family=sm.families.Binomial())
    model_results = model.fit()

    print(model_results.summary())
    print("R2:", r2_score(y_true=endog_data, y_pred=model_results.fittedvalues))

    figsize = (4, 2.7)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = normal_size

    plt.figure()
    plt.hexbin(endog_data, model_results.fittedvalues, bins="log")
    plt.xlabel("real data")
    plt.ylabel("prediction")
    plt.colorbar()

    filename = os.path.expanduser(os.path.join("~/tmp/to_send", "hexbin_GLM1" + ".pdf"))
    print("Saving to {}".format(filename))
    plt.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    # Data generation.
    global_mask = burned_area_cube.data.mask

    # Predicted burned area values.
    ba_predicted = np.zeros_like(global_mask, dtype=np.float64)
    ba_predicted[~global_mask] = model_results.fittedvalues
    ba_predicted = np.ma.MaskedArray(ba_predicted, mask=global_mask)

    # Observed values.
    ba_data = np.zeros_like(global_mask, dtype=np.float64)
    ba_data[~global_mask] = endog_data.values
    ba_data = np.ma.MaskedArray(ba_data, mask=global_mask)

    # Plotting of burned area data & predictions:
    #  - ba_predicted: predicted burned area
    #  - ba_data: observed
    #  - model_name: Name for titles AND filenames
    model_name = "GLMv1"
    figs = map_model_output(
        ba_predicted, ba_data, model_name, normal_size, normal_coast_linewidth
    )

    figsize = (5, 3)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = normal_size

    columns = list(map(map_name, exog_data2.columns))

    def get_trim_func(n=10, cont_str="..."):
        def trim(string):
            if len(string) > n:
                string = string[: n - len(cont_str)]
                string += cont_str
            return string

        return trim

    # https://stackoverflow.com/questions/55289921/matplotlib-matshow-xtick-labels-on-top-and-bottom/55289968
    n = len(columns)

    fig, ax = plt.subplots()

    corr_arr = np.ma.MaskedArray(exog_data2.corr().values)
    corr_arr.mask = np.zeros_like(corr_arr)
    # Ignore diagnals, since they will all be 1 anyway!
    np.fill_diagonal(corr_arr.mask, True)

    im = ax.matshow(corr_arr, interpolation="none")

    fig.colorbar(im, pad=0.04, shrink=0.95)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(map(get_trim_func(), columns))
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(columns)

    # Set ticks on top of axes on
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    # Rotate and align bottom ticklabels
    # plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
    #          ha="right", va="center", rotation_mode="anchor")
    # Rotate and align top ticklabels
    plt.setp(
        [tick.label2 for tick in ax.xaxis.get_major_ticks()],
        rotation=45,
        ha="left",
        va="center",
        rotation_mode="anchor",
    )

    # For some reason the code below does not work and produces overallping top labels.
    # Maybe needed to set the rotation_mode, ha, and va parameters from above?
    # plt.xticks(range(len(columns)), map(get_trim_func(), columns), rotation=45)
    # plt.yticks(range(len(columns)), columns)
    # plt.colorbar(pad=0.2)

    # ax.set_title("Correlation Matrix", pad=40)
    fig.tight_layout()

    filename = os.path.expanduser(
        os.path.join("~/tmp/to_send", "correlation_GLM1" + ".pdf")
    )
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    print(model_results.summary().as_latex())

    # # Random Forest
    # ## Using same data as for the GLMv1 above

    regr = RandomForestRegressor(n_estimators=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        exog_data2, endog_data, random_state=1, shuffle=True, test_size=0.3
    )
    regr.fit(X_train, y_train)
    print("R2 train:", regr.score(X_train, y_train))
    print("R2 test:", regr.score(X_test, y_test))

    global_mask = burned_area_cube.data.mask

    ba_predicted = np.zeros_like(global_mask, dtype=np.float64)
    ba_predicted[~global_mask] = regr.predict(exog_data2)
    ba_predicted = np.ma.MaskedArray(ba_predicted, mask=global_mask)

    ba_data = np.zeros_like(global_mask, dtype=np.float64)
    ba_data[~global_mask] = endog_data
    ba_data = np.ma.MaskedArray(ba_data, mask=global_mask)

    # Plotting of burned area data & predictions:
    #  - ba_predicted: predicted burned area
    #  - ba_data: observed
    #  - model_name: Name for titles AND filenames
    model_name = "RFv1"
    figs = map_model_output(
        ba_predicted, ba_data, model_name, normal_size, normal_coast_linewidth
    )

    mpl.rcParams["figure.figsize"] = (20, 12)
    mpl.rcParams["font.size"] = 18
    fig, axes = partial_dependence_plot(
        regr,
        X_test,
        X_test.columns,
        n_cols=4,
        grid_resolution=70,
        coverage=0.05,
        predicted_name="burned area",
    )
    plt.subplots_adjust(wspace=0.16)
    _ = list(ax.axes.get_yaxis().set_ticks([]) for ax in axes)

    filename = os.path.expanduser(os.path.join("~/tmp/to_send", "pdp_RFv1" + ".pdf"))
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    ################ DOES NOT DEMONSTRATE ANYTHING!! #################
    lims = []
    varnames = []
    for ax in axes:
        lims.append(ax.get_ylim()[1])
        varnames.append(ax.get_xlabel())
    for i, j in zip(varnames, exog_data2.columns.values):
        assert i == j

    tvalues = np.abs(model_results.tvalues)
    plt.figure(figsize=(10, 10))
    plt.plot(lims, tvalues, linestyle="", marker="o")

    for i, txt in enumerate(varnames):
        plt.annotate(txt, (lims[i], tvalues[i]))

    plt.xlabel("RF")
    plt.ylabel("GLM T Value")
    plt.show()

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
    forest_names = list(map(map_name, exog_data2.columns.values))
    importances_df = pd.DataFrame(
        {
            "Name": forest_names,
            "Importance": importances,
            "Importance STD": std,
            "Ratio": np.array(std) / np.array(importances),
        }
    )
    print(
        importances_df.sort_values("Importance", ascending=False).to_latex(
            index=False, float_format="{:0.3f}".format
        )
    )

    # # Creating backup slides
    # ## First the datasets that were simply selected
    # ## Then the datasets after they were modified with NN interpolation and isolated outlier removal

    figsize = (5, 3.8)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = normal_size

    for cube in lat_land_datasets.cubes:
        fig = cube_plotting(
            cube,
            log=log_map(cube.name()),
            coastline_kwargs={"linewidth": normal_coast_linewidth},
        )
        filename = os.path.expanduser(
            os.path.join(
                "~/tmp/to_send", "backup_" + cube.name().replace(" ", "_") + ".pdf"
            )
        )
        print("Saving to {}".format(filename))
        fig.savefig(
            filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
        )

    for cube in filled_datasets.cubes:
        fig = cube_plotting(
            cube,
            log=log_map(cube.name()),
            coastline_kwargs={"linewidth": normal_coast_linewidth},
        )
        filename = os.path.expanduser(
            os.path.join(
                "~/tmp/to_send", "backup_mod_" + cube.name().replace(" ", "_") + ".pdf"
            )
        )
        print("Saving to {}".format(filename))
        fig.savefig(
            filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
        )
