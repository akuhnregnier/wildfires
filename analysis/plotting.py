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

    # Plotting differences.

    # https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html
    # Use log-modulus transformation

    perc_diffs = (ba_data - ba_predicted) / ba_data

    fig = cube_plotting(
        perc_diffs,
        cmap="viridis",
        title="({}) Perc. Diff. (Observed - Predicted)".format(model_name),
        coastline_kwargs={"linewidth": coast_linewidth},
    )
    figs.append(fig)
    return figs


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
    auto_log_title=False,
    transform_vmin_vmax=False,
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
        transform_vmin_vmax (bool): If `transform_vmin_vmax` and `log`, apply the log
            function used to transform the data to `vmin` and `vmax` (in `kwargs`)
            as well.
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

        if vmin is not None and vmax is not None and vmin > vmax:
            raise ValueError(f"vmin ({vmin}) must not be larger than vmax ({vmax}).")

        n_bins = kwargs.get("nbins", 10)

        if log:
            boundaries = []

            pos_range = None
            neg_range = None

            mask = cube.data > 0
            if np.any(mask):
                sel_data = cube.data[mask]
                pos_range = [
                    math.floor(math.log10(np.min(sel_data))),
                    math.ceil(math.log10(np.max(sel_data))),
                ]

            mask = cube.data < 0
            if np.any(mask):
                sel_data = cube.data[mask]
                neg_range = [
                    math.ceil(math.log10(np.max(np.abs(sel_data)))),
                    math.floor(math.log10(np.min(np.abs(sel_data)))),
                ]

            neg_bins = 0
            pos_bins = 0

            if not log_auto_bins:
                # If there is both positive and negative data.
                if pos_range and neg_range:
                    ratio = (np.ptp(pos_range) + 1) / (
                        np.ptp(pos_range) + np.ptp(neg_range) + 2
                    )
                    neg_bins = math.floor((n_bins + 1) * (1 - ratio))
                    pos_bins = math.ceil((n_bins + 1) * ratio)
                    # TODO: Best way to handle vmin/vmax? Trim values adjacent to 0 instead?
                    if vmin is not None:
                        if vmin < 0:
                            neg_bins -= 1
                            neg_range[0] = math.log10(-vmin)
                        else:
                            neg_bins = 0
                            neg_range = None

                            pos_bins -= 1
                            pos_range[0] = math.log10(vmin)
                    if vmax is not None:
                        if vmax >= 0:
                            pos_bins -= 1
                            pos_range[1] = math.log10(vmax)
                        else:
                            pos_bins = 0
                            pos_range = None

                            neg_bins -= 1
                            neg_range[1] = math.log10(-vmax)

                # If there is only positive data.
                elif pos_range:
                    pos_bins = n_bins + 1
                    # TODO: Best way to handle vmin/vmax? Trim values adjacent to 0 instead?
                    if vmin is not None:
                        pos_bins -= 1
                        if vmin < 0:
                            pos_bins -= 1
                            neg_bins = 1
                            neg_range[0] = math.log10(-vmin)
                            neg_range[1] = neg_range[0]
                        else:
                            pos_range[0] = math.log10(vmin)
                    if vmax is not None:
                        if vmax >= 0:
                            pos_bins -= 1
                            pos_range[1] = math.log10(vmax)
                        else:
                            pos_bins = 0
                            pos_range = None

                            neg_bins = 1
                            neg_range[1] = math.log10(-vmax)
                            neg_range[0] = neg_range[1]

                # If there is only negative data.
                elif neg_range:
                    neg_bins = n_bins + 1
                    # TODO: Best way to handle vmin/vmax? Trim values adjacent to 0 instead?
                    if vmin is not None:
                        if vmin < 0:
                            neg_bins -= 1
                            neg_range[0] = math.log10(-vmin)
                        else:
                            neg_bins = 0
                            neg_range = None

                            pos_bins = 1
                            pos_range[0] = math.log10(vmin)
                            pos_range[1] = pos_range[0]
                    if vmax is not None:
                        neg_bins -= 1
                        neg_range[1] = vmax
                        if vmax >= 0:
                            neg_bins -= 1
                            pos_bins = 1
                            pos_range[1] = math.log10(vmax)
                            pos_range[0] = pos_range[1]
                        else:
                            neg_range[1] = math.log10(-vmax)

            if log_auto_bins:
                if vmin is not None:
                    boundaries.append(vmin)
                    if vmin < 0:
                        neg_range[0] = math.floor(math.log10(-vmin) - 1)
                    else:
                        neg_range = None

                        pos_range[0] = math.ceil(math.log10(vmin))

                if vmax is not None:
                    boundaries.append(vmax)
                    if vmax >= 0:
                        pos_range[1] = math.floor(math.log10(vmax))
                    else:
                        pos_range = None

                        neg_range[1] = math.ceil(math.log10(-vmax))

                if neg_range:
                    neg_bins = np.ptp(neg_range) + 1
                if pos_range:
                    pos_bins = np.ptp(pos_range) + 1

                if neg_bins > 0:
                    boundaries.extend(
                        -np.logspace(neg_range[0], neg_range[1], neg_bins)
                    )

                if neg_bins > 0 and pos_bins > 0 or np.any(np.isclose(cube.data, 0)):
                    boundaries.append(0)

                if pos_bins > 0:
                    boundaries.extend(np.logspace(pos_range[0], pos_range[1], pos_bins))

                boundaries = list(set(boundaries))
                boundaries.sort()
            else:
                if neg_bins > 0:
                    boundaries.extend(
                        -np.logspace(neg_range[0], neg_range[1], neg_bins)
                    )

                if neg_bins > 0 and pos_bins > 0 or np.any(np.isclose(cube.data, 0)):
                    boundaries.append(0)
                    pos_bins -= 1

                if pos_bins > 0:
                    boundaries.extend(np.logspace(pos_range[0], pos_range[1], pos_bins))

            if not neg_bins and not pos_bins:
                raise ValueError("No data found.")

        else:
            data_min = np.min(cube.data)
            data_max = np.max(cube.data)
            lin_min = data_min if vmin is None else vmin
            lin_max = data_max if vmax is None else vmax

            boundaries = np.linspace(lin_min, lin_max, n_bins + 1)

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
        nrows=int(math.ceil(len(features)) / n_cols), ncols=n_cols, squeeze=False
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


def test_pdp():
    import string

    class A:
        def predict(self, X):
            # Prevent modification from persisting.
            X = X.copy()
            X.iloc[:, 0] *= -1
            return np.sum(X, axis=1)

    test_data = np.array([[1, 1], [2, 2]])

    model = A()
    X = pd.DataFrame(
        test_data, columns=list(string.ascii_lowercase)[: test_data.shape[1]]
    )
    features = list(string.ascii_lowercase[: test_data.shape[1]])

    fig, axes = partial_dependence_plot(
        model, X, features, grid_resolution=2, norm_y_ticks=False, percentiles=(0, 1)
    )
    plt.show()


def test_plotting():
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


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)
    # plt.close('all')
    test_plotting()
