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
from matplotlib.colors import from_levels_and_colors
from tqdm import tqdm

from wildfires.data.datasets import dummy_lat_lon_cube
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)


class FigureSaver:
    """Saving figures.

    If `debug`, `debug_options` will be used. Otherwise `options` are used.

    Use like to save created figure(s) automatically:

    with FigureSaver("filename"):
        plt.figure()
        ...

    with FigureSaver(("filename", "filename2")):
        plt.figure()
        ...
        plt.figure()
        ...

    with FigureSaver() as saver:
        fig = plt.figure()
        ...
        saver.save_figure(fig, "plot_name")

    """

    debug = False
    directory = "."

    # These options serve as default value that may be overridden during
    # initialisation.
    options = {
        "bbox_inches": "tight",
        "transparent": True,
        "rasterised": True,
        "filetype": "pdf",
        "dpi": 600,
    }

    debug_options = {
        "bbox_inches": "tight",
        "transparent": True,
        "rasterised": True,
        "filetype": "png",
        "dpi": 200,
    }

    def __init__(self, filename=None, directory=None, debug=None, **kwargs):
        """Initialise figure saver.

        Args:
            filename ((iterable of) str or None): If None, disable automatic saving. Otherwise the
                number of strings passed must match the number of opened figures at
                the termination of the context manager.
            directory ((iterable of) str or None): The directory to save figures in.
                If None, use the class default.
            debug (bool or None): If None, use the class default.

        """
        if debug is None:
            # Reset to default.
            self.debug = type(self).debug
        else:
            self.debug = debug

        if directory is None:
            # Reset to default.
            self.directory = type(self).directory
        else:
            self.directory = directory

        if filename is None:
            # Disable automatic saving.
            logger.debug("Automatic saving disabled.")
            self.filenames = None
            self.directories = None
        else:
            self.filenames = (filename,) if isinstance(filename, str) else filename
            self.directories = (
                (self.directory,) if isinstance(self.directory, str) else self.directory
            )

            if len(self.directories) != 1 and len(self.directories) != len(
                self.filenames
            ):
                raise ValueError(
                    "If more than one directory is given, the number of given directories "
                    "has to match the number of given file names. Got "
                    f"{len(self.directories)} directories and {len(self.filenames)} "
                    "file names."
                )
            if len(self.directories) == 1:
                self.directories = [self.directories[0]] * len(self.filenames)

        self.options = self.debug_options.copy() if self.debug else self.options.copy()
        self.options.update(kwargs)

        self.suffix = (
            self.options["filetype"]
            if "." in self.options["filetype"]
            else "." + self.options["filetype"]
        )
        del self.options["filetype"]

    def __enter__(self):
        self.old_fignums = plt.get_fignums()
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return False  # Re-raise exception.
        new_figure_numbers = plt.get_fignums()
        if new_figure_numbers == self.old_fignums:
            raise RuntimeError("No new figures detected.")

        fignums_save = [
            num for num in new_figure_numbers if num not in self.old_fignums
        ]

        if len(fignums_save) != len(self.filenames):
            raise RuntimeError(
                f"Expected {len(self.filenames)} figures, but got {len(fignums_save)}"
            )

        saved_figures = [
            num
            if not plt.figure(num).get_label()
            else (num, plt.figure(num).get_label())
            for num in fignums_save
        ]

        logger.debug(f"Saving figures {saved_figures}.")

        for fignum, directory, filename in zip(
            fignums_save, self.directories, self.filenames
        ):
            fig = plt.figure(fignum)
            self.save_figure(fig, filename, directory)

    def save_figure(self, fig, filename, directory=None):
        if directory is None:
            # Use default.
            directory = type(self).directory

        filepath = (
            os.path.expanduser(
                os.path.abspath(os.path.expanduser(os.path.join(directory, filename)))
            )
            + self.suffix
        )

        logger.debug("Saving figure to '{}'.".format(filepath))

        fig.savefig(filepath, **self.options)


def get_cubes_vmin_vmax(cubes, vmin_vmax_percentiles=(0.0, 100.0)):
    """Get vmin and vmax from a list of cubes given two percentiles.

    Args:
        cubes (iris.cube.CubeList): List of cubes.
        vmin_vmax_percentiles (tuple or None): The two percentiles, used to set the minimum
            and maximum values on the colorbar. If `None`, use the minimum and maximum
            of the data (equivalent to percentiles of (0, 100)).
    Returns:
        tuple: tuple of floats (vmin, vmax) if `vmin_vmax_percentiles` is not (0, 100) in which case
            (None, None) will be returned.

    """
    if vmin_vmax_percentiles is None or np.all(
        np.isclose(np.array(vmin_vmax_percentiles), np.array([0, 100]))
    ):
        return None, None

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

    output = []

    if np.isclose(vmin_vmax_percentiles[0], 0):
        output.append(None)
    else:
        output.append(min(limit[0] for limit in limits))

    if np.isclose(vmin_vmax_percentiles[1], 100):
        output.append(None)
    else:
        output.append(max(limit[1] for limit in limits))

    return output


def map_model_output(ba_predicted, ba_data, model_name, coast_linewidth):
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
    mpl.rcParams["figure.figsize"] = figsize

    vmin = min((np.min(ba_predicted), np.min(ba_data)))
    vmax = max((np.max(ba_predicted), np.max(ba_data)))

    # Plotting predicted.
    fig = cube_plotting(
        ba_predicted,
        cmap="Reds",
        label="Fraction",
        title="Predicted Mean Burned Area ({})".format(model_name),
        log=True,
        coastline_kwargs={"linewidth": coast_linewidth},
        vmin=vmin,
        vmax=vmax,
    )
    figs.append(fig)

    # Plotting observed.
    fig = cube_plotting(
        ba_data,
        cmap="Reds",
        label="Fraction",
        title="Mean observed burned area (GFEDv4)",
        log=True,
        coastline_kwargs={"linewidth": coast_linewidth},
        vmin=vmin,
        vmax=vmax,
    )
    figs.append(fig)

    # Plotting differences.

    # https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html
    # Use log-modulus transformation

    perc_diffs = (ba_data - ba_predicted) / ba_data

    fig = cube_plotting(
        perc_diffs,
        cmap="viridis",
        title="({}) Perc. Diff. (Observed - Predicted)".format(model_name),
        coastline_kwargs={"linewidth": coast_linewidth},
        log=True,
    )
    figs.append(fig)
    return figs


def get_bin_edges(
    data=None,
    vmin=None,
    vmax=None,
    n_bins=11,
    log=True,
    min_edge_type="manual",
    min_edge=None,
):
    """Get bin edges.

    Args:
        data (numpy.ndarray): Data array to determine limits.
        vmin (float): Minimum bin edge.
        vmax (float): Maximum bin edge.
        n_bins (int or str): If "auto" (only applies when `log=True`), determine bin
            edges using the data (see `min_edge`).
        log (bool): If `log`, bin edges are computed in log space (base 10).
        min_edge_type (str): If "manual", the supplied `min_edge` will be used for the
            minimum exponent (for both positive and negative edges), and `vmin` and
            `vmax` for the upper limits. If "auto", determine `min_edge` from the
            data. If "symmetric", determine `min_edge` from the data, but use the same
            value for the positive and negative edges. If either `vmin` or `vmax` is
            very close to (or equal to) 0, `min_edge` is also required as the starting
            point of the bin edges (see examples).
        min_edge (float or None): If None, the minimum (absolute) data value will be
            used. Otherwise the supplied float will be used to set the minimum
            exponent of the log bins.

    Returns:
        list: The bin edges.

    TODO:
        Make sure the n_bins parameter is respected, which is not true at the moment
        for cases where vmin or vmax are close to 0, or when an additional 0 is
        inserted otherwise.

    Examples:
        >>> get_bin_edges(vmin=0, vmax=100, n_bins=2, log=False)
        [0.0, 50.0, 100.0]
        >>> get_bin_edges(vmin=1, vmax=100, n_bins=2, log=True, min_edge=1)
        [1.0, 10.0, 100.0]
        >>> get_bin_edges(vmin=-100, vmax=100, n_bins="auto", log=True, min_edge=1,
        ...               min_edge_type="manual")
        [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
        >>> get_bin_edges(vmin=-100, vmax=1000, n_bins="auto", log=True, min_edge=1,
        ...               min_edge_type="manual")
        [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0]
        >>> get_bin_edges(vmin=0, vmax=100, n_bins=2, log=True, min_edge=1)
        [0.0, 1.0, 10.0, 100.0]

    """
    input_args = locals()
    if not log:
        assert type(n_bins) == int
        if any(vlim is None for vlim in (vmin, vmax)) and data is None:
            raise ValueError("Need data for vmin/vmax that are None.")

    if vmin is not None and vmax is not None and vmin > vmax:
        raise ValueError(f"vmin ({vmin}) must not be larger than vmax ({vmax}).")

    if log and vmin is not None and vmax is not None and data is None:
        assert (
            min_edge is not None
        ), "When not supplying data, `min_edge` needs to be given."
        assert (
            min_edge_type == "manual"
        ), "When not supplying data, `min_edge_type` needs to be 'manual'."
        # Get bin edges without relying on data at all.
        input_args["data"] = np.array([1])
        return get_bin_edges(**input_args)

    min_bin = vmin if vmin is not None else np.min(data)
    max_bin = vmax if vmax is not None else np.max(data)

    n_close_lim = sum(np.isclose(lim_bin, 0) for lim_bin in (min_bin, max_bin))
    assert n_close_lim <= 1, "At most one limit should be close to 0."

    min_force = True if vmin is not None else False
    max_force = True if vmax is not None else False

    if not log:
        assert (
            type(n_bins) == int
        ), f"Bin number must be an integer if `log=False`. Got {repr(n_bins)} instead."
        return list(np.linspace(min_bin, max_bin, n_bins + 1))

    # Only the case with log=True remains here.
    # Positive and negative data are handled the same way (after taking the absolute
    # value) and the resulting bins are then stitched together if needed.

    zero_mask = ~np.isclose(data, 0)
    split_data = (data[(data > 0) & zero_mask], np.abs(data[(data < 0) & zero_mask]))
    multipliers = (1, -1)

    if min_bin >= 0:
        split_data = split_data[:1]
        multipliers = (1,)
    if max_bin <= 0:
        split_data = split_data[1:]
        multipliers = (-1,)

    if n_close_lim == 1:
        assert (
            len(split_data) == 1
        ), "There should be only 1 split dataset if one of the limits is 0."

    limits_list = []
    force_bins_list = []

    if min_edge_type == "manual":
        assert min_edge is not None, "Need valid `min_edge` for 'manual' edge type."

    if min_edge is not None:
        if min_edge_type != "manual":
            raise ValueError(
                "Value for `min_edge` supplied even though `min_edge_type` was set to "
                f"'{min_edge_type}'."
            )
        min_edge_type = "manual"

    if min_edge_type in ("symmetric", "auto"):
        if min_edge is not None:
            raise ValueError(
                "Value for `min_edge` supplied even though "
                f"`min_edge_type` was set to '{min_edge_type}'."
            )
        if min_edge_type == "symmetric":
            min_edge_type = "manual"
            min_edge = np.min(np.abs(data))
        else:
            # If edge type is auto, data is needed to infer the bounds.
            if not all(map(np.any, split_data)):
                raise ValueError(
                    f"Insufficient data for `min_edge_type` '{min_edge_type}'."
                )

    fallback_min_edge = 1e-7
    if min_edge < 1e-8:
        logger.warning(
            f"Low `min_edge` ('{min_edge}') replaced with '{fallback_min_edge}'."
        )
        min_edge = 1e-7

    for s_data, multiplier in zip(split_data, multipliers):
        # Get the relevant limits for the data in both the positive and negative case.
        # Do this by checking the sign and magnitude of the predefined limits, and
        # using max/min of the selected data if necessary.
        limits = []
        force_bins = []
        for s_bin, func, force in zip(
            (min_bin, max_bin)[::multiplier],
            (np.min, np.max),
            (min_force, max_force)[::multiplier],
        ):
            force_bin = False
            # If vmin/vmax are applicable.
            if multiplier * s_bin >= 0:
                limits.append(multiplier * s_bin)
                if force:
                    force_bin = True
            # Purely data-informed - relies on respective data being available.
            elif min_edge_type == "manual":
                limits.append(min_edge)
            elif min_edge_type == "auto":
                limits.append(func(s_data))
            else:
                raise ValueError(f"Unknown `min_edge_type` '{min_edge_type}'.")

            force_bins.append(force_bin)

        limits_list.append(limits)
        force_bins_list.append(force_bins)

    log_limits_list = []
    for limits in limits_list:
        log_limits = []
        if np.isclose(limits[0], 0):
            assert min_edge is not None
            to_process = min_edge
        else:
            to_process = limits[0]
        log_limits.append(math.floor(np.log10(to_process)))
        log_limits.append(math.ceil(np.log10(limits[1])))
        log_limits_list.append(log_limits)

    boundaries = []
    if n_bins != "auto":
        if len(split_data) == 2:
            denom = sum([np.ptp(log_limits) + 1 for log_limits in log_limits_list])
            ratio = (np.ptp(log_limits_list[0]) + 1) / (denom)
            split_bins = (
                math.ceil((n_bins + 1) * ratio),
                math.floor((n_bins + 1) * (1 - ratio)),
            )
        else:
            split_bins = (n_bins + 1,)

        for multiplier, limits, bin_number in zip(multipliers, limits_list, split_bins):
            if np.isclose(limits[0], 0):
                limits[0] = min_edge
                boundaries.append(0)
            boundaries.extend(multiplier * np.geomspace(*limits, bin_number))
    else:
        for multiplier, force_bins, limits, log_limits in zip(
            multipliers, force_bins_list, limits_list, log_limits_list
        ):
            # Forced bins are kept as explicit limits. If the bin is not forced, the
            # value derived from the log limits is used.
            boundaries.extend(
                multiplier * limit
                for forced, limit in zip(force_bins, limits)
                if forced
            )

            force_exponent_mod = np.array([1, -1])
            force_exponent_mod[~np.array(force_bins, dtype=np.bool_)] = 0
            log_limits = tuple(np.array(log_limits) + force_exponent_mod)
            limit_diff = log_limits[1] - log_limits[0]
            if limit_diff > 0 or np.isclose(limit_diff, 0):
                boundaries.extend(
                    multiplier * np.logspace(*log_limits, np.ptp(log_limits) + 1)
                )

    # TODO: Remove something to make up for this?
    if len(split_data) == 2:
        boundaries.append(0)

    return sorted(float(edge) for edge in boundaries)


def cube_plotting(
    cube,
    log=False,
    rasterized=True,
    coastline_kwargs={},
    dummy_lat_lims=(-90, 90),
    dummy_lon_lims=(-180, 180),
    vmin_vmax_percentiles=(0, 100),
    projection=ccrs.Robinson(),
    animation_output=False,
    ax=None,
    mesh=None,
    new_colorbar=True,
    title_text=None,
    transform_vmin_vmax=False,
    nbins=10,
    log_auto_bins=True,
    **kwargs,
):
    """Pretty plotting.

    Eg. for temperature, use
    cmap='Reds'
    label=r"T ($\degree$C)"

    Args:
        cube: Cube to plot.
        log: True to log.
        rasterized: Rasterize pcolormesh (but not the text).
        dummy_lat_lims: Tuple passed to dummy_lat_lon_cube function in case the input
            argument is not a cube.
        dummy_lon_lims: Tuple passed to dummy_lat_lon_cube function in case the input
            argument is not a cube.
        vmin_vmax_percentiles (tuple or None): The two percentiles, used to set the minimum
            and maximum values on the colorbar. If `None`, use the minimum and maximum
            of the data (equivalent to percentiles of (0, 100)). Explicitly passed-in
            `vmin` and `vmax` parameters take precedence.
        projection: A projection as defined in `cartopy.crs`.
        animation_output (bool): Output additional variables required to create an
            animation.
        ax (matplotlib axis): Axis to plot onto.
        mesh (matplotlib.collections.QuadMesh): If given, update the mesh instead of
            creating a new one.
        new_colorbar (bool): If True, create a new colorbar. Turn off for animation.
        title_text (matplotlib.text.Text): Title text.
        transform_vmin_vmax (bool): If `transform_vmin_vmax` and `log`, apply the log
            function used to transform the data to `vmin` and `vmax` (in `kwargs`)
            as well.
        nbins (int): Number of bins. Does not apply if `log` and `log_auto_bins`.
        log_auto_bins (bool): Make log bins stick to integers.
        possible kwargs:
            title: str or None of False. If None or False, no title will be plotted.
            cmap: Example: 'viridis', 'Reds', 'Reds_r', etc... Can also be a
                `matplotlib.colors.Colormap` instance.
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
                    - format

    """
    if not isinstance(cube, iris.cube.Cube):
        cube = dummy_lat_lon_cube(
            cube, lat_lims=dummy_lat_lims, lon_lims=dummy_lon_lims
        )

    cube = cube.copy()

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection=projection)
    else:
        fig = ax.get_figure()

    if mesh is not None:
        mesh.set_array(cube.data.ravel())
    else:
        for coord_name in ["latitude", "longitude"]:
            if not cube.coord(coord_name).has_bounds():
                cube.coord(coord_name).guess_bounds()

        gridlons = cube.coord("longitude").contiguous_bounds()
        gridlats = cube.coord("latitude").contiguous_bounds()

        data_vmin, data_vmax = get_cubes_vmin_vmax([cube], vmin_vmax_percentiles)

        vmin = kwargs.get("vmin", data_vmin)
        vmax = kwargs.get("vmax", data_vmax)

        boundaries = get_bin_edges(
            cube.data,
            vmin,
            vmax,
            "auto" if log and log_auto_bins else nbins,
            log,
            "symmetric",
        )

        if vmin is None and vmax is None:
            extend = "neither"
            n_colors = len(boundaries) - 1
        if vmin is not None and vmax is None:
            extend = "min"
            n_colors = len(boundaries)
        if vmax is not None and vmin is None:
            extend = "max"
            n_colors = len(boundaries)
        if vmin is not None and vmax is not None:
            extend = "both"
            n_colors = len(boundaries) + 1

        if n_colors > plt.get_cmap(kwargs.get("cmap", "viridis")).N:
            logger.warning(
                f"Expected at most {plt.get_cmap(kwargs.get('cmap', 'viridis')).N} "
                f"colors, but got {n_colors}."
            )

        cmap, norm = from_levels_and_colors(
            boundaries,
            plt.get_cmap(kwargs.get("cmap", "viridis"))(np.linspace(0, 1, n_colors)),
            extend=extend,
        )

        # This forces data points 'exactly' (very close) to the upper limit of the last
        # bin to be recognised as part of that bin, as opposed to out of bounds.
        if vmin is None and vmax is None:
            norm.clip = True

        mesh = ax.pcolormesh(
            gridlons,
            gridlats,
            cube.data,
            # cmap=kwargs.get("cmap", "viridis"),
            cmap=cmap,
            norm=norm,
            # NOTE: This transform here may differ from the projection argument.
            transform=ccrs.PlateCarree(),
            rasterized=rasterized,
            # TODO: FIXME: Setting vmin/vmax to something close to the data extremes seems
            # to mess up norm / cmap... - Ignore for now??
            vmin=vmin,
            vmax=vmax,
        )

    ax.coastlines(resolution="110m", **coastline_kwargs)

    colorbar_kwargs = {
        "label": kwargs.get("label", str(cube.units)),
        "orientation": kwargs.get("orientation", "vertical"),
        "fraction": kwargs.get("fraction", 0.15),
        "pad": kwargs.get("pad", 0.07),
        "shrink": kwargs.get("shrink", 0.9)
        if kwargs.get("orientation", "vertical") == "horizontal"
        else kwargs.get("shrink", 0.7),
        "aspect": kwargs.get("aspect", 30),
        "anchor": kwargs.get("anchor", (0.5, 1.0)),
        "panchor": kwargs.get("panchor", (0.5, 0.0)),
        "format": "%.1e" if log else None,
    }
    # TODO: https://matplotlib.org/3.1.0/gallery/axes_grid1/simple_colorbar.html
    # TODO: Use this to attach the colorbar to the axes, not the figure. Errors were
    # encountered due to cartopy geoaxes.
    if new_colorbar:
        fig.colorbar(mesh, **colorbar_kwargs)
    title = kwargs.get("title", cube.name())
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
    n_cols="auto",
    prefer="columns",
    grid_resolution=100,
    percentiles=(0.0, 1.0),
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
        n_cols (int or str): Number of columns in the resulting plot. If 'auto',
            try to produce a square grid. Preference will be given to more rows or
            more columns based on `prefer`.
        prefer (str): If 'rows', prefer more rows. If 'columns', prefer more columns.
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
            averaged_predictions.append(np.mean(model.predict(calc_X)))

        datasets.append(np.array(averaged_predictions).reshape(-1, 1))

    datasets = np.hstack(datasets)
    results = pd.DataFrame(datasets, columns=features)

    valid = ("rows", "columns")
    if prefer not in valid:
        raise ValueError(
            f"Unknown parameter value '{prefer}' for `prefer`."
            f"Choose one of '{valid}'."
        )

    if prefer == "rows":
        n_cols = int(math.floor(np.sqrt(len(features))))
    else:
        n_cols = int(math.ceil(np.sqrt(len(features))))

    fig, axes = plt.subplots(
        nrows=math.ceil(len(features) / n_cols), ncols=n_cols, squeeze=False
    )

    axes = axes.flatten()

    if norm_y_ticks:
        predicted_name = "relative " + predicted_name
        # Calculate the normalised ranges.
        results -= results.to_numpy().min()
        results /= results.to_numpy().max()

    for (i, (ax, feature)) in enumerate(zip(axes, features)):
        ax.plot(quantile_data[feature], results[feature])
        ax.set_xlabel(feature)
        if i % n_cols == 0:
            ax.set_ylabel(predicted_name)

    # Make empty plots (should they exist) invisible.
    for ax in axes[len(features) :]:
        ax.set_axis_off()

    # TODO: Make the positioning and the number of labels more uniform.
    # if norm_y_ticks:
    #     y_ticklabels = []
    #     for ax in axes:
    #         y_ticklabels.extend(ax.get_yticks())
    #     y_tick_values = np.array(y_ticklabels)
    #     min_val = np.min(y_tick_values)
    #     max_val = np.max(y_tick_values - min_val)
    #     # for ax in axes:
    #     #     # ax.set_yticks((ax.get_yticks() - min_val) / max_val)
    #     #     ax.set_yticks([])
    #     for ax in axes:
    #         ticks = ax.get_yticks().tolist()
    #         ticks = ["test" for tick in ticks]
    #         ax.set_yticklabels(ticks)

    plt.tight_layout()

    return fig, axes


def sample_pdp():
    import string

    class A:
        def predict(self, X):
            # Prevent modification from persisting.
            X = X.copy()
            X.iloc[:, 0] *= -1
            return np.sum(X, axis=1)

    test_data = np.zeros((2, 18)) + np.array([[1], [2]])

    model = A()
    X = pd.DataFrame(
        test_data, columns=list(string.ascii_lowercase)[: test_data.shape[1]]
    )
    features = list(string.ascii_lowercase[: test_data.shape[1]])

    fig, axes = partial_dependence_plot(
        model, X, features, grid_resolution=2, norm_y_ticks=False, percentiles=(0, 1)
    )
    plt.show()


def sample_plotting():
    np.random.seed(1)
    cube = dummy_lat_lon_cube(
        ((-4 + np.arange(12).reshape(4, 3))) ** 5 + np.random.random((4, 3)), units="T"
    )
    cube_plotting(
        cube,
        log=True,
        title="Testing",
        orientation="horizontal",
        nbins=9,
        log_auto_bins=True,
        cmap="brewer_RdYlBu_11",
        # vmin=None,
        vmin=-500,
        vmax=50,
    )
    plt.show()


def sample_map_model_output():
    data = np.random.normal(size=(100, 50))
    data2 = np.random.normal(size=(100, 50))
    map_model_output(data, data2, "testing", 1.0)


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)
