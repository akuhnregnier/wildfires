#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting functions."""
import logging
import logging.config
import math
import os

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iris.time import PartialDateTime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from wildfires.analysis.processing import log_modulus
from wildfires.data.datasets import dummy_lat_lon_cube
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)


def map_model_output(ba_predicted, ba_data, model_name, textsize, coast_linewidth):
    """Plotting of burned area data & predictions.

    Args:
        ba_predicted: predicted burned area
        ba_data: observed
        model_name (str): Name of the run.

    Returns:
        tuple: The created figures.

    """
    figs = []
    # Plotting params.
    figsize = (4, 2.7)
    dpi = 600
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["font.size"] = textsize

    log_vmin = min((np.min(np.log(ba_predicted)), np.min(np.log(ba_data))))
    log_vmax = max((np.max(np.log(ba_predicted)), np.max(np.log(ba_data))))

    # Plotting predicted.
    fig = cube_plotting(
        ba_predicted,
        cmap="Reds",
        label="ln(Fraction)",
        title="Predicted Mean Burned Area ({})".format(model_name),
        log=True,
        coastline_kwargs={"linewidth": coast_linewidth},
        vmin=log_vmin,
        vmax=log_vmax,
    )
    figs.append(fig)

    filename = os.path.expanduser(
        os.path.join("~/tmp/to_send", "predicted_burned_area_" + model_name + ".pdf")
    )
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    # Plotting observed.
    fig = cube_plotting(
        ba_data,
        cmap="Reds",
        label="ln(Fraction)",
        title="Mean observed burned area (GFEDv4)",
        log=True,
        coastline_kwargs={"linewidth": coast_linewidth},
        vmin=log_vmin,
        vmax=log_vmax,
    )
    figs.append(fig)

    filename = os.path.expanduser(
        os.path.join(
            "~/tmp/to_send", "mean_observed_burned_area_" + model_name + ".pdf"
        )
    )
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )

    # Plotting differences.

    # https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html
    # Use log-modulus transformation

    perc_diffs = (ba_data - ba_predicted) / ba_data
    log_mod_diffs = log_modulus(perc_diffs)

    fig = cube_plotting(
        log_mod_diffs,
        cmap="viridis",
        label=r"$\mathrm{sign}(x) \times \ln(|x| + 1)$",
        title="({}) log-modulus (Observed - Predicted)".format(model_name),
        coastline_kwargs={"linewidth": coast_linewidth},
    )
    figs.append(fig)

    filename = os.path.expanduser(
        os.path.join("~/tmp/to_send", "difference_burned_area_" + model_name + ".pdf")
    )
    print("Saving to {}".format(filename))
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True, rasterised=True
    )
    return figs


def cube_plotting(
    cube,
    log=False,
    rasterized=True,
    coastline_kwargs={},
    dummy_lat_lims=(-90, 90),
    dummy_lon_lims=(-180, 180),
    **kwargs
):
    """Pretty plotting.

    For temperature, use
    cmap='Reds'
    label=r"T ($\degree$C)"

    Args:
        cube: Cube to plot.
        log: True to log.
        rasterized: Rasterize pcolormesh (but not the text)
        dummy_lat_lims: Tuple passed to dummy_lat_lon_cube function in case the input
            argument is not a cube
        dummy_lon_lims: Tuple passed to dummy_lat_lon_cube function in case the input
            argument is not a cube

        possible kwargs:
            title: str or None of False. If None or False, no title will be plotted.
            cmap: Example: 'viridis', 'Reds', 'Reds_r', etc...
            vmin: Minimum value for colorbar.
            vmax: Maximum value for colorbar.
            colorbar_kwargs: Dictionary with any number of the following keys and
                corresponding values:
                    - label
                    - orientation
                    - fraction
                    - pad
                    - srhink
                    - aspect
                    - anchor
                    - panchor

    """
    if not isinstance(cube, iris.cube.Cube):
        cube = dummy_lat_lon_cube(
            cube, lat_lims=dummy_lat_lims, lon_lims=dummy_lon_lims
        )

    cube = cube.copy()

    if log:
        # print("Logging cube")
        if not cube.long_name:
            cube.long_name = cube.name()
        future_name = "log " + cube.long_name
        cube = iris.analysis.maths.log(cube)
        cube.long_name = future_name

    for coord_name in ["latitude", "longitude"]:
        if not cube.coord(coord_name).has_bounds():
            cube.coord(coord_name).guess_bounds()

    gridlons = cube.coord("longitude").contiguous_bounds()
    gridlats = cube.coord("latitude").contiguous_bounds()

    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    # print("Requesting data to plot")
    plt.pcolormesh(
        gridlons,
        gridlats,
        cube.data,
        cmap=kwargs.get("cmap", "viridis"),
        transform=ccrs.PlateCarree(),
        rasterized=rasterized,
        vmin=kwargs.get("vmin"),
        vmax=kwargs.get("vmax"),
    )
    ax.coastlines(**coastline_kwargs)

    colorbar_kwargs = {
        "label": kwargs.get("label", str(cube.units)),
        "orientation": kwargs.get("orientation", "horizontal"),
        "fraction": kwargs.get("fraction", 0.15),
        "pad": kwargs.get("pad", 0.07),
        "shrink": kwargs.get("shrink", 0.9),
        "aspect": kwargs.get("aspect", 30),
        "anchor": kwargs.get("anchor", (0.5, 1.0)),
        "panchor": kwargs.get("panchor", (0.5, 0.0)),
    }

    plt.colorbar(**colorbar_kwargs)
    title = kwargs.get("title", cube.name())
    if title:
        plt.title(title)
    return fig


def partial_dependence_plot(
    model,
    X,
    features,
    n_cols=3,
    grid_resolution=100,
    percentiles=(0.05, 0.95),
    coverage=1,
    random_state=None,
    predicted_name="burned area",
    norm_y_ticks=False,
):
    """Plot 1 dimensional partial dependence plots.

    Args:
        model: A fitted model with a 'predict' method.
        X (pandas.DataFrame): Dataframe containing the data over which the
            partial dependence plots are generated.
        features (list): A list of int or string with which to select
            columns from X. The type must match the column labels.
        n_cols (int): Number of columns in the resulting plot.
        grid_resolution (int): Number of points to plot between the
            specified percentiles.
        percentiles (tuple of float): Percentiles between which to plot the
            pdp.
        coverage (float): A float between 0 and 1 which dictates what
            fraction of data is used to generate the plots.
        random_state (int or None): Used to call np.random.seed(). This is
            only used if coverage < 1. Supplying None causes the random
            number generator to initialise itself using the system clock
            (or using something similar which will not be constant from one
            run to the next).
        predicted_name (string): Name on the y axis.
        norm_y_ticks (bool): If True, scale y ticks using the range.

    Returns:
        fig, ax: Figure and axes holding the pdp.

    """
    if norm_y_ticks:
        predicted_name = "relative " + predicted_name
    features = list(features)

    quantiles = X[features].quantile(percentiles)
    quantile_data = pd.DataFrame(
        np.linspace(quantiles.iloc[0], quantiles.iloc[1], grid_resolution),
        columns=features,
    )

    datasets = []

    if not np.isclose(coverage, 1):
        logger.debug("Selecting subset of data with coverage:{:}".format(coverage))
        np.random.seed(random_state)

        # Select a random subset of the data.
        permuted_indices = np.random.permutation(np.arange(X.shape[0]))
        permuted_indices = permuted_indices[: int(coverage * X.shape[0])]
    else:
        logger.debug("Selecting all data.")
        # Select all of the data.
        permuted_indices = np.arange(X.shape[0])

    X_selected = X.iloc[permuted_indices]

    for feature in tqdm(features):
        series = quantile_data[feature]

        # for each possible value of the selected feature, average over all
        # possible combinations of the other features
        averaged_predictions = []
        calc_X = X_selected.copy()
        for value in series:
            calc_X[feature] = value
            predictions = model.predict(calc_X)
            averaged_predictions.append(np.mean(predictions))

        datasets.append(np.array(averaged_predictions).reshape(-1, 1))

    datasets = np.hstack(datasets)
    results = pd.DataFrame(datasets, columns=features)

    fig, axes = plt.subplots(
        nrows=int(math.ceil(float(len(features)) / n_cols)), ncols=n_cols, squeeze=False
    )

    axes = axes.flatten()

    for (i, (ax, feature)) in enumerate(zip(axes, features)):
        ax.plot(quantile_data[feature], results[feature])
        ax.set_xlabel(feature)
        if i % n_cols == 0:
            ax.set_ylabel(predicted_name)

    # Make empty plots (should they exist) invisible.
    for ax in axes[len(features) :]:
        ax.set_axis_off()

    # TODO:
    # if norm_y_ticks:
    #     y_ticklabels = []
    #     for ax in axes:
    #         y_tick_values.extend(ax.get_yticks())
    #     y_tick_values = np.array(y_tick_values)
    #     min_val = np.min(y_tick_values)
    #     max_val = np.max(y_tick_values - min_val)
    #     for ax in axes:
    #         ax.set_yticks((ax.get_yticks() - min_val) / max_val)

    plt.tight_layout()

    return fig, axes


if __name__ == "__main__":
    import string

    logging.config.dictConfig(LOGGING)

    m = 10000
    n = 20

    class A:
        def predict(self, X):
            X.iloc[:, 0] *= -1
            return np.sum(X, axis=1)

    model = A()
    X = pd.DataFrame(np.random.random((m, n)), columns=list(string.ascii_lowercase)[:n])
    features = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    fig, axes = partial_dependence_plot(model, X, features, grid_resolution=100)
