#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data analysis."""
import logging
import logging.config
import math
from functools import partial, reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from wildfires.analysis.plotting import (
    FigureSaver,
    cube_plotting,
    map_model_output,
    partial_dependence_plot,
)
from wildfires.analysis.processing import log_map, map_name, vif
from wildfires.data.cube_aggregation import (
    IGNORED_DATASETS,
    get_all_datasets,
    prepare_selection,
)
from wildfires.data.datasets import dataset_times
from wildfires.logging_config import LOGGING
from wildfires.utils import get_masked_array, get_unmasked
from wildfires.utils import land_mask as get_land_mask

logger = logging.getLogger(__name__)


def print_vifs(exog_data, thres=6):
    vifs = vif(exog_data)
    print(vifs.to_string(index=False, float_format="{:0.1f}".format))

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


def print_importances(regr):
    """Print RF importances.

    Args:
        regr (RandomForestRegressor):

    """
    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
    forest_names = list(map(map_name, exog_data.columns.values))
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


def plot_histograms(datasets):
    """Plot histograms of datasets."""
    # TODO: Allow plotting of only a subset of datasets.
    n_plots = len(datasets.cubes)
    n_cols = int(math.ceil(np.sqrt(n_plots)))

    fig, axes = plt.subplots(
        nrows=int(math.ceil(n_plots / n_cols)), ncols=n_cols, squeeze=False
    )
    axes = axes.flatten()
    for (i, (ax, feature)) in enumerate(zip(axes, range(n_plots))):
        ax.hist(get_unmasked(datasets.cubes[feature].data), density=True, bins=70)
        ax.set_xlabel(datasets.pretty_variable_names[feature])
        ax.set_yscale("log")

    for ax in axes[n_plots:]:
        ax.set_axis_off()

    plt.tight_layout()


def TripleFigureSaver(model_name, *args, **kwargs):
    """Plotting of burned area data & predictions:
        - ba_predicted: predicted burned area
        - ba_data: observed
        - model_name: Name for titles AND filenames

    """
    return FigureSaver(
        filename=(
            "predicted_burned_area_" + model_name,
            "mean_observed_burned_area_" + model_name,
            "difference_burned_area_" + model_name,
        ),
        *args,
        **kwargs,
    )


def print_dataset_times(datasets, latex=False):
    """Print information about the dataset times to stdout."""
    times_df = dataset_times(datasets.datasets)[2]
    if times_df is not None:
        if latex:
            print(times_df.to_latex(index=False))
        else:
            print(times_df.to_string(index=False))
    else:
        print("No time information found.")


def data_processing(
    selection,
    target_variable="monthly burned area",
    which="monthly",
    transformations=None,
    deletions=None,
    log_var_names=None,
    sqrt_var_names=None,
    verbose=True,
):
    """Create datasets for further analysis and model fitting."""
    if verbose:
        selection.show("pretty")
        print_dataset_times(selection, latex=False)

    raw_datasets = prepare_selection(selection, which=which)

    # Get land mask.
    # TODO: Check that this works consistently, then it can be removed.
    assert 1440 == raw_datasets.cubes[0].coord("longitude").points.shape[0]
    land_mask = ~get_land_mask(
        n_lon=raw_datasets.cubes[0].coord("longitude").points.shape[0]
    )

    # Define a latitude mask which ignores data beyond 60 degrees, as GSMaP data does
    # not extend to those latitudes.
    # TODO: Instead of `mean_datasets` use a more generic name to take into account
    # different `which` arguments for `prepare_selection()`.
    lats = raw_datasets.cubes[0].coord("latitude").points
    lons = raw_datasets.cubes[0].coord("longitude").points

    latitude_grid = np.meshgrid(lats, lons, indexing="ij")[0]
    lat_mask = np.abs(latitude_grid) > 60

    # Apply masks.

    # Make a deep copy so that the original cubes are preserved.
    masked_datasets = raw_datasets.copy(deep=True)

    masks_to_apply = (lat_mask, land_mask)
    for cube in masked_datasets.cubes:
        cube.data.mask |= reduce(np.logical_or, masks_to_apply)

    # Filling/processing/cleaning datasets.

    # TODO: Make this kind of processing internal to each Dataset.
    sif_cube = masked_datasets.select_variables("SIF", inplace=False).cube
    invalid_mask = np.logical_or(sif_cube.data.data > 20, sif_cube.data.data < 0)
    logger.info(f"Masking {np.sum(invalid_mask)} invalid values for SIF.")
    sif_cube.data.mask |= invalid_mask

    # TODO: Make this kind of processing internal to each Dataset.
    lightning_cube = masked_datasets.select_variables(
        "Combined Flash Rate Monthly Climatology", inplace=False
    ).cube
    invalid_mask = lightning_cube.data.data < 0
    logger.info(f"Masking {np.sum(invalid_mask)} invalid values for LIS/OTD.")
    lightning_cube.data.mask |= invalid_mask

    filled_datasets = masked_datasets.copy(deep=True).fill(land_mask, lat_mask)

    # Creating exog and endog pandas containers.
    burned_area_cube = filled_datasets.select_variables(
        target_variable, inplace=False
    ).cube
    endog_data = pd.Series(get_unmasked(burned_area_cube.data))
    master_mask = burned_area_cube.data.mask

    exog_datasets = filled_datasets.remove_variables(target_variable, inplace=False)
    data = []
    for cube in exog_datasets.cubes:
        data.append(get_unmasked(cube.data).reshape(-1, 1))

    exog_data = pd.DataFrame(
        np.hstack(data), columns=exog_datasets.pretty_variable_names
    )

    if transformations is not None:
        for new_var, processing_func in transformations.items():
            exog_data[new_var] = processing_func(exog_data)
    if deletions is not None:
        for delete_var in deletions:
            del exog_data[delete_var]

    if verbose:
        print("Names before:")
        print(exog_data.columns)

    # Carry out log transformation for select variables.
    if log_var_names is not None:
        for name in log_var_names:
            mod_data = exog_data[name] + 0.01
            assert np.all(mod_data >= (0.01 - 1e-8)), "{:}".format(name)
            exog_data["log " + name] = np.log(mod_data)
            del exog_data[name]

    # Carry out square root transformation
    if sqrt_var_names is not None:
        for name in sqrt_var_names:
            assert np.all(exog_data[name] >= 0), "{:}".format(name)
            exog_data["sqrt " + name] = np.sqrt(exog_data[name])
            del exog_data[name]

    if verbose:
        print("Names after:")
        print(exog_data.columns)

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


def corr_plot(exog_data):
    columns = list(map(map_name, exog_data.columns))

    def trim(string, n=10, cont_str="..."):
        if len(string) > n:
            string = string[: n - len(cont_str)]
            string += cont_str
        return string

    # https://stackoverflow.com/questions/55289921/matplotlib-matshow-xtick-labels-on-top-and-bottom/55289968
    n = len(columns)
    fig, ax = plt.subplots()

    corr_arr = np.ma.MaskedArray(exog_data.corr().values)
    corr_arr.mask = np.zeros_like(corr_arr)
    # Ignore diagnals, since they will all be 1 anyway!
    np.fill_diagonal(corr_arr.mask, True)

    im = ax.matshow(corr_arr, interpolation="none")

    fig.colorbar(im, pad=0.04, shrink=0.95)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(map(partial(trim, n=10), columns))
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

    # For some reason the code below does not work and produces overlapping top labels.
    # Maybe needed to set the rotation_mode, ha, and va parameters from above?
    # plt.xticks(range(len(columns)), map(get_trim_func(), columns), rotation=45)
    # plt.yticks(range(len(columns)), columns)
    # plt.colorbar(pad=0.2)

    # ax.set_title("Correlation Matrix", pad=40)
    fig.tight_layout()


def GLM(endog_data, exog_data):
    model = sm.GLM(endog_data, exog_data, family=sm.families.Binomial())
    model_results = model.fit()

    print(model_results.summary())
    print("R2:", r2_score(y_true=endog_data, y_pred=model_results.fittedvalues))

    mpl.rcParams["figure.figsize"] = (4, 2.7)

    with FigureSaver("hexbin_GLM1"):
        plt.figure()
        plt.hexbin(endog_data, model_results.fittedvalues, bins="log")
        plt.xlabel("real data")
        plt.ylabel("prediction")
        plt.colorbar()

    # Data generation.

    # Predicted burned area values.
    ba_predicted = get_masked_array(model_results.fittedvalues, master_mask)

    # Observed values.
    ba_data = get_masked_array(endog_data.values, master_mask)

    # Plotting of burned area data & predictions:
    #  - ba_predicted: predicted burned area
    #  - ba_data: observed
    #  - model_name: Name for titles AND filenames
    model_name = "GLMv1"
    with TripleFigureSaver(model_name):
        figs = map_model_output(
            ba_predicted, ba_data, model_name, normal_coast_linewidth
        )

    print(model_results.summary().as_latex())
    return model_results


def RF(endog_data, exog_data):
    regr = RandomForestRegressor(n_estimators=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        exog_data, endog_data, random_state=1, shuffle=True, test_size=0.3
    )
    regr.fit(X_train, y_train)
    print("R2 train:", regr.score(X_train, y_train))
    print("R2 test:", regr.score(X_test, y_test))

    ba_predicted = get_masked_array(regr.predict(exog_data), master_mask)

    ba_data = get_masked_array(endog_data, master_mask)

    model_name = "RFv1"
    with TripleFigureSaver(model_name):
        figs = map_model_output(
            ba_predicted, ba_data, model_name, normal_coast_linewidth
        )

    mpl.rcParams["figure.figsize"] = (20, 12)
    mpl.rcParams["font.size"] = 18

    with FigureSaver("pdp_RFv1"):
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

    return regr


if __name__ == "__main__":
    # General setup.
    logging.config.dictConfig(LOGGING)

    FigureSaver.directory = "~/tmp/to_send"
    FigureSaver.debug = True

    # TODO: Plotting setup in a more rigorous manner.
    normal_coast_linewidth = 0.5
    mpl.rcParams["font.size"] = 9.0
    verbose = True

    target_variable = "monthly burned area"

    # Creation of new variables.
    transformations = {
        "temperature range": lambda exog_data: (
            exog_data["maximum temperature"] - exog_data["minimum temperature"]
        )
    }
    # Variables to be deleted after the aforementioned transformations.
    deletions = ("minimum temperature",)

    # Carry out transformations, replacing old variables in the process.
    log_var_names = ["temperature range", "dry_days", "dry_day_period"]
    sqrt_var_names = ["Combined Flash Rate Monthly Climatology", "popd"]

    # Dataset selection.
    selection = get_all_datasets(ignore_names=IGNORED_DATASETS)
    selection.remove_datasets("GSMaP Dry Day Period")
    selection = selection.select_variables(
        [
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
    )
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = data_processing(
        selection,
        which="mean",
        transformations=transformations,
        deletions=deletions,
        log_var_names=log_var_names,
        sqrt_var_names=sqrt_var_names,
    )

    # Plotting land mask.
    with FigureSaver("land_mask"):
        mpl.rcParams["figure.figsize"] = (8, 5)
        fig = cube_plotting(land_mask.astype("int64"), title="Land Mask", cmap="Reds_r")

    # Plot histograms.
    mpl.rcParams["figure.figsize"] = (20, 12)
    plot_histograms(masked_datasets)

    # Plotting masked burned area and AGB.
    cube = masked_datasets.select_variables("monthly burned area", inplace=False).cube
    with FigureSaver(cube.name().replace(" ", "_")):
        mpl.rcParams["figure.figsize"] = (5, 3.8)
        fig = cube_plotting(
            cube,
            cmap="Reds",
            log=True,
            label="ln(Fraction)",
            title="Log Mean Burned Area (GFEDv4)",
            coastline_kwargs={"linewidth": normal_coast_linewidth},
        )

    cube = masked_datasets.select_variables("AGBtree", inplace=False).cube
    with FigureSaver(cube.name().replace(" ", "_")):
        mpl.rcParams["figure.figsize"] = (4, 2.7)
        fig = cube_plotting(
            cube,
            cmap="viridis",
            log=True,
            label=r"kg m$^{-2}$",
            title="AGBtree",
            coastline_kwargs={"linewidth": normal_coast_linewidth},
        )

    # Plotting of filled/processed Mask and AGB.
    with FigureSaver("land_mask2"):
        mpl.rcParams["figure.figsize"] = (8, 5)
        fig = cube_plotting(
            filled_datasets.select_variables(
                "monthly burned area", strict=True, inplace=False
            ).cube.data.mask.astype("int64"),
            title="Land Mask",
            cmap="Reds_r",
        )

    cube = filled_datasets.select_variables("AGBtree", inplace=False).cube
    with FigureSaver(cube.name().replace(" ", "_") + "_interpolated"):
        mpl.rcParams["figure.figsize"] = (4, 2.7)
        cube_plotting(
            cube,
            cmap="viridis",
            log=True,
            label=r"kg m$^{-2}$",
            title="AGBtree (interpolated)",
            coastline_kwargs={"linewidth": normal_coast_linewidth},
        )

    # Start of variable analysis!!

    print_vifs(exog_data, thres=6)

    with FigureSaver("correlation"):
        mpl.rcParams["figure.figsize"] = (5, 3)
        corr_plot(exog_data)

    # Creating backup slides.

    # First the original datasets.
    mpl.rcParams["figure.figsize"] = (5, 3.8)

    for cube in masked_datasets.cubes:
        with FigureSaver("backup_" + cube.name().replace(" ", "_")):
            fig = cube_plotting(
                cube,
                log=log_map(cube.name()),
                coastline_kwargs={"linewidth": normal_coast_linewidth},
            )

    # Then the datasets after they were processed.
    for cube in filled_datasets.cubes:
        with FigureSaver("backup_mod_" + cube.name().replace(" ", "_")):
            fig = cube_plotting(
                cube,
                log=log_map(cube.name()),
                coastline_kwargs={"linewidth": normal_coast_linewidth},
            )

    # Model Fitting.
    model_results = GLM(endog_data, exog_data)
    regr = RF(endog_data, exog_data)

    if verbose:
        print_importances(regr)
