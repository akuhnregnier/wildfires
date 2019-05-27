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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from wildfires.analysis.processing import log_modulus
from wildfires.data.datasets import dummy_lat_lon_cube
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)


def get_cubes_vmin_vmax(cubes, vmin_vmax_percentiles=(10, 90)):
    """Get vmin and vmax from a list of cubes given two percentiles.

    Args:
        cubes (iris.cube.CubeList): List of cubes.
        vmin_vmax_percentiles (tuple or None): The two percentiles, used to set the minimum
            and maximum values on the colorbar. If `None`, use the minimum and maximum
            of the data (equivalent to percentiles of (0, 100)). Explicitly passed-in
            `vmin` and `vmax` parameters take precedence.

    Returns:
        tuple of float: (vmin, vmax).

    """
    if vmin_vmax_percentiles is None:
        vmin_vmax_percentiles = (0, 100)
    limits = []
    for cube in cubes:
        if isinstance(cube.data, np.ma.core.MaskedArray):
            if isinstance(cube.data.mask, np.ndarray):
                valid_data = cube.data.data[~cube.data.mask]
            elif cube.data.mask:
                raise ValueError("All data is masked.")
            else:
                valid_data = cube.data.data
        else:
            valid_data = cube.data

        limits.append(np.percentile(valid_data, vmin_vmax_percentiles))

    return min(limit[0] for limit in limits), max(limit[1] for limit in limits)


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
    vmin_vmax_percentiles=(10, 90),
    projection=ccrs.Robinson(),
    animation_output=False,
    ax=None,
    mesh=None,
    new_colorbar=True,
    title_text=None,
    auto_log_title=False,
    **kwargs,
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
        vmin_vmax_percentiles (tuple or None): The two percentiles, used to set the minimum
            and maximum values on the colorbar. If `None`, use the minimum and maximum
            of the data (equivalent to percentiles of (0, 100)). Explicitly passed-in
            `vmin` and `vmax` parameters take precedence.
        projection: A projection as defined in `cartopy.crs`.
        animation_output (bool): Output additional variables required to create an
            animation.
        ax (matplotlib axis):
        mesh (matplotlib.collections.QuadMesh): If given, update the mesh instead of
            creating a new one.
        new_colorbar (bool): If True, create a new colorbar. Turn off for animation.
        title_text (matplotlib.text.Text): Title text.
        auto_log_title (bool): If `auto_log_title`, prepend "log " to `title` if
            `log`.

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

    data_vmin, data_vmax = get_cubes_vmin_vmax([cube], vmin_vmax_percentiles)

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection=projection)
    else:
        fig = ax.get_figure()

    if mesh is not None:
        mesh.set_array(cube.data.ravel())
    else:
        mesh = ax.pcolormesh(
            gridlons,
            gridlats,
            cube.data,
            cmap=kwargs.get("cmap", "viridis"),
            # NOTE: This transform here may differ from the projection argument.
            transform=ccrs.PlateCarree(),
            rasterized=rasterized,
            vmin=kwargs.get("vmin", data_vmin),
            vmax=kwargs.get("vmax", data_vmax),
        )

    ax.coastlines(resolution="110m", **coastline_kwargs)

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
    if new_colorbar:
        fig.colorbar(mesh, **colorbar_kwargs)
    title = kwargs.get("title", cube.name())
    if log and auto_log_title and kwargs.get("title"):
        title = "log " + title
    if title:
        if isinstance(title, mpl.text.Text):
            title_text.set_text(title)
        else:
            title_text = fig.suptitle(title)

    if animation_output:
        return fig, ax, mesh, title_text
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
