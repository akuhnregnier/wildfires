#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting functions."""
import logging
import logging.config
import math
import os
import warnings

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import from_levels_and_colors
from tqdm import tqdm

from ..data import dummy_lat_lon_cube
from ..logging_config import LOGGING
from ..utils import (
    in_360_longitude_system,
    multiline,
    select_valid_subset,
    translate_longitude_system,
)

__all__ = (
    "FigureSaver",
    "cube_plotting",
    "get_bin_edges",
    "get_cubes_vmin_vmax",
    "get_pdp_data",
    "map_model_output",
    "partial_dependence_plot",
)

logger = logging.getLogger(__name__)


class PlottingError(Exception):
    """Base class for exception in the plotting module."""


class MaskedDataError(PlottingError):
    """Raised when trying to plot fully-masked data."""


class FigureSaver:
    """Save figures using pre-defined options and directories.

    If `debug`, `debug_options` will be used. Otherwise `options` are used.

    Figure(s) are saved automatically:

    with FigureSaver("filename"):
        plt.figure()
        ...

    with FigureSaver(("filename1", "filename2")):
        plt.figure()
        ...
        plt.figure()
        ...

    Manual saving using the defined options is also possible.

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
        "transparent": False,
        "rasterised": True,
        "filetype": "png",
        "dpi": 350,
    }

    def __init__(self, filenames=None, *, directories=None, debug=None, **kwargs):
        """Initialise figure saver.

        The initialised FigureSaver instance can be used repeatedly (as a context
        manager) to save figures by calling it with at least filenames and optionally
        directories, debug state, and saving options.

        Args:
            filenames ((iterable of) str or None): If None, the FigureSaver instance
                must be called with a list of filenames and used as a context manager
                for automatic saving. Otherwise, the number of strings passed must
                match the number of opened figures at the termination of the context
                manager.
            directory ((iterable of) str or None): The directory to save figures in.
                If None, use `FigureSaver.directory`. New directories will be created if
                they do not exist.
            debug (bool or None): Select the pre-set settings with which figures will
                be saved. If None, use `FigureSaver.debug`.
            kwargs: Optional kwargs which are passed to plt.savefig().

        """
        # Backwards compatibility.
        if "filename" in kwargs:
            warnings.warn(
                "The `filename` argument is deprecated in favour of the `filenames` "
                "argument, which takes precedence.",
                FutureWarning,
            )
            # Only use the deprecated argument if the updated version is not used.
            if filenames is None:
                filenames = kwargs.pop("filename")
        if "directory" in kwargs:
            warnings.warn(
                "The `directory` argument is deprecated in favour of the `directories` "
                "argument, which takes precedence.",
                FutureWarning,
            )
            # Only use the deprecated argument if the updated version is not used.
            if directories is None:
                directories = kwargs.pop("directory")

        # Set instance defaults.
        if debug is not None:
            self.debug = debug

        directories = directories if directories is not None else self.directory

        self.directories = (
            (directories,) if isinstance(directories, str) else directories
        )

        if filenames is not None:
            self.filenames = (filenames,) if isinstance(filenames, str) else filenames

            if len(self.directories) != 1 and len(self.directories) != len(
                self.filenames
            ):
                raise ValueError(
                    multiline(
                        f"""If multiple directories are given, their number has to match
                        the number of file names, but got {len(self.directories)}
                        directories and {len(self.filenames)} file names."""
                    )
                )

        # Make sure to resolve the home directory.
        self.directories = tuple(map(os.path.expanduser, self.directories))

        self.options = self.debug_options.copy() if self.debug else self.options.copy()
        self.options.update(kwargs)

    def __call__(self, filenames=None, sub_directory=None):
        """Return a copy containing the given filenames for figure saving.

        An optional sub-directory can also be specified for figures saved by the
        returned object.

        This is meant to be used as a context manager:
            >>> figure_saver = FigureSaver(**options)  # doctest: +SKIP
            >>> with figure_saver("filename"):  # doctest: +SKIP
            ...     plt.figure()  # doctest: +SKIP

        Directories, options, etc... which the FigureSaver instance was initialised
        with will be used to save the figures.

        Args:
            filenames ((iterable of) str): Filenames used to save created figures.
            sub_directory (str): If given, figures will be saved in a sub-directory
                `sub_directory` of the pre-specified directory/directories.

        """
        new_inst = type(self)(
            filenames,
            directories=[os.path.join(orig, sub_directory) for orig in self.directories]
            if sub_directory is not None
            else self.directories,
            debug=self.debug,
        )
        new_inst.options = self.options
        return new_inst

    @property
    def suffix(self):
        return (
            self.options["filetype"]
            if "." in self.options["filetype"]
            else "." + self.options["filetype"]
        )

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
                f"Expected {len(self.filenames)} figures, but got {len(fignums_save)}."
            )

        saved_figures = [
            num
            if not plt.figure(num).get_label()
            else (num, plt.figure(num).get_label())
            for num in fignums_save
        ]

        logger.debug(f"Saving figures {saved_figures}.")

        if len(self.directories) == 1:
            # Adapt to the number of figures.
            directories = [self.directories[0]] * len(self.filenames)
        else:
            directories = self.directories

        for fignum, directory, filename in zip(
            fignums_save, directories, self.filenames
        ):
            fig = plt.figure(fignum)
            self.save_figure(fig, filename, directory)

    def save_figure(self, fig, filename, directory=None, sub_directory=None):
        """Save a single figure.

        Args:
            fig (matplotlib.figure.Figure): Figure to save.
            filename (str): Filename where the figure will be saved.
            directory (str): Directory to save the figure in.
            sub_directory (str): If given, figures will be saved in a sub-directory
                `sub_directory` of the pre-specified directory/directories.

        Raises:
            ValueError: If multiple default directories were specified and no explicit
                directory is supplied here. Since only one figure is being saved here,
                it is unclear which directory to choose. In this case, the context
                manager interface is to be used.

        """
        if directory is None:
            if len(self.directories) > 1:
                raise ValueError("More than 1 default directory specified.")
            # Use default.
            directory = self.directories[0]
        if sub_directory is not None:
            directory = os.path.join(directory, sub_directory)

        os.makedirs(directory, exist_ok=True)

        filepath = (
            os.path.expanduser(
                os.path.abspath(os.path.expanduser(os.path.join(directory, filename)))
            )
            + self.suffix
        )

        logger.debug("Saving figure to '{}'.".format(filepath))

        fig.savefig(
            filepath,
            **{
                option: value
                for option, value in self.options.items()
                if option != "filetype"
            },
        )


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
    figsize = (5, 3.33)
    mpl.rcParams["figure.figsize"] = figsize

    vmin = min((np.min(ba_predicted), np.min(ba_data)))
    vmax = max((np.max(ba_predicted), np.max(ba_data)))

    boundaries = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # Plotting predicted.
    fig = cube_plotting(
        ba_predicted,
        cmap="brewer_RdYlBu_11_r",
        label="Burnt Area Fraction",
        title=None,
        log=True,
        coastline_kwargs={"linewidth": coast_linewidth},
        vmin=vmin,
        vmax=vmax,
        min_edge=vmin,
        extend="min",
        boundaries=boundaries,
    )
    figs.append(fig)

    # Plotting observed.
    fig = cube_plotting(
        ba_data,
        cmap="brewer_RdYlBu_11_r",
        label="Burnt Area Fraction",
        title=None,
        log=True,
        coastline_kwargs={"linewidth": coast_linewidth},
        vmin=vmin,
        vmax=vmax,
        min_edge=vmin,
        extend="min",
        boundaries=boundaries,
    )
    figs.append(fig)

    # Plotting differences.

    # https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html
    # (Do not!) use log-modulus transformation

    perc_diffs = (ba_data - ba_predicted) / ba_data

    diff_boundaries = [-1e3, -1e0, -1e-1, -1e-2, 0, 1e-1, 5e-1, 1e0]

    fig = cube_plotting(
        perc_diffs,
        cmap="brewer_RdYlBu_11_r",
        title=None,
        coastline_kwargs={"linewidth": coast_linewidth},
        log=True,
        boundaries=diff_boundaries,
        extend="min" if np.max(perc_diffs) <= max(diff_boundaries) else "both",
        label="(Observed - Predicted) / Observed",
    )
    figs.append(fig)
    return figs


def _get_log_bin_edges(vmin, vmax, n_bins):
    if isinstance(n_bins, (int, np.integer)):
        return np.geomspace(vmin, vmax, n_bins + 1)
    else:
        edges = []
        # "Auto" bins.
        vmax_log = np.log10(vmax)
        vmin_log = np.log10(vmin)
        # Convert to float here explicitly as np.float32 does not have an
        # `is_integer()` method, for example.
        if not float(vmax_log).is_integer():
            edges.append(vmax)
            vmax_log = math.floor(vmax_log)
        # To ensure it is an integer (round() does not turn a numpy float into an int).
        vmax_log = int(round(vmax_log))

        if not float(vmin_log).is_integer():
            edges.append(vmin)
            vmin_log = math.ceil(vmin_log)
        # To ensure it is an integer (round() does not turn a numpy float into an int).
        vmin_log = int(round(vmin_log))

        edges.extend(np.logspace(vmin_log, vmax_log, vmax_log - vmin_log + 1))
        return sorted(edges)


def get_bin_edges(
    data=None,
    vmin=None,
    vmax=None,
    n_bins=11,
    log=True,
    min_edge_type="manual",
    min_edge=None,
    simple_lin_bins=False,
):
    """Get bin edges.

    Args:
        data (numpy.ndarray or None): Data array to determine limits.
        vmin (float or None): Minimum bin edge.
        vmax (float or None): Maximum bin edge.
        n_bins (int or str): If "auto" (only applies when `log=True`), determine bin
            edges using the data (see `min_edge`).
        log (bool): If `log`, bin edges are computed in log space (base 10).
        min_edge_type (str): If "manual", the supplied `min_edge` will be used for the
            minimum edge(s), and `vmin` and `vmax` for the upper limits. If "auto",
            determine `min_edge` from the data. If "symmetric", determine `min_edge`
            from the data, but use the same value for the positive and negative edges.
            If either `vmin` or `vmax` is very close to (or equal to) 0, `min_edge` is
            also required as the starting point of the bin edges (see examples).
        min_edge (float, iterable of float, or None): If None, the minimum (absolute)
            data value will be used. Otherwise the supplied float will be used to set
            the minimum exponent of the log bins. If two floats are given, they will
            be used for the positive and negative ranges, respectively.
        simple_lin_bins (bool): If True, simply create `n_bins` divisions from vmin
            (minimum data) to vmax (maximum data). If False (default), explicitly
            structure the bins around 0 if needed. If there is only positive or only
            negative data, the two cases are equivalent.

    Returns:
        list: The bin edges.

    Examples:
        >>> get_bin_edges(vmin=0, vmax=100, n_bins=2, log=False)
        [0.0, 50.0, 100.0]
        >>> get_bin_edges(vmin=1, vmax=100, n_bins=2, log=True, min_edge=1)
        [1.0, 10.0, 100.0]
        >>> get_bin_edges(vmin=-100, vmax=100, n_bins="auto", log=True, min_edge=1,
        ...               min_edge_type="manual")
        [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
        >>> get_bin_edges(vmin=-150, vmax=150, n_bins="auto", log=True, min_edge=1,
        ...               min_edge_type="manual")
        [-150.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 150.0]
        >>> get_bin_edges(vmin=-1000, vmax=100, n_bins=7, log=True, min_edge=1,
        ...               min_edge_type="manual")
        [-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
        >>> get_bin_edges(vmin=-100, vmax=1000, n_bins="auto", log=True, min_edge=1,
        ...               min_edge_type="manual")
        [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0]
        >>> get_bin_edges(vmin=0, vmax=100, n_bins=3, log=True, min_edge=1)
        [0.0, 1.0, 10.0, 100.0]
        >>> get_bin_edges(np.array([0, 1000]), n_bins=4, min_edge=1.0, log=True)
        [0.0, 1.0, 10.0, 100.0, 1000.0]
        >>> get_bin_edges(np.array([0, 1, 1000]), n_bins=4, min_edge_type="auto", log=True)
        [0.0, 1.0, 10.0, 100.0, 1000.0]
        >>> get_bin_edges(np.array([0, 10, 100]), n_bins=2, min_edge_type="auto", log=True)
        [0.0, 10.0, 100.0]
        >>> get_bin_edges(vmin=-20, vmax=80, n_bins=5, log=False,
        ...               simple_lin_bins=True)
        [-20.0, 0.0, 20.0, 40.0, 60.0, 80.0]
        >>> get_bin_edges(vmin=-20, vmax=80, n_bins=5, log=False,
        ...               simple_lin_bins=False)
        [-20.0, 0.0, 20.0, 40.0, 60.0, 80.0]
        >>> np.all(
        ...     np.isclose(
        ...         get_bin_edges(
        ...             vmin=-20,
        ...             vmax=77,
        ...             n_bins=9,
        ...             log=False,
        ...             simple_lin_bins=True,
        ...         ),
        ...         np.linspace(-20, 77, 10),
        ...     )
        ... )
        True
        >>> get_bin_edges(vmin=-20, vmax=77, n_bins=9, log=False,
        ...               simple_lin_bins=False)
        [-20.0, -10.0, 0.0, 11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]

    """
    if not log:
        assert isinstance(
            n_bins, (int, np.integer)
        ), f"Bin number must be an integer if `log=False`. Got {repr(n_bins)} instead."
        if any(vlim is None for vlim in (vmin, vmax)) and data is None:
            raise ValueError("Need data when vmin & vmax are not supplied (are None).")

    if all(vlim is not None for vlim in (vmin, vmax)) and vmin > vmax:
        raise ValueError(f"vmin ({vmin}) must not be larger than vmax ({vmax}).")

    if log and all(vlim is not None for vlim in (vmin, vmax)) and data is None:
        assert (
            min_edge is not None
        ), "When not supplying data, `min_edge` needs to be given."
        assert (
            min_edge_type == "manual"
        ), "When not supplying data, `min_edge_type` needs to be 'manual'."

    vmin = vmin if vmin is not None else np.min(data)
    vmax = vmax if vmax is not None else np.max(data)

    n_close_lim = sum(np.isclose(vlim, 0) for vlim in (vmin, vmax))
    assert n_close_lim <= 1, "At most one limit should be close to 0."

    if not log:
        if simple_lin_bins or not (vmin < 0 and vmax > 0):
            # Handle simple cases where the bin edges do not cross 0.
            return list(np.linspace(vmin, vmax, n_bins + 1))
        else:
            vrange = vmax - vmin

            pos_edges = math.ceil((n_bins + 1) * np.abs(vmin) / vrange)
            neg_edges = math.ceil((n_bins + 1) * vmax / vrange)

            # The desired number of edges is bins + 2, since we are removing one edge
            # (the duplicated 0) and thus we are left with bins + 1, the number of
            # edges required to form 'bins' number of bins as desired.
            if pos_edges + neg_edges != n_bins + 2:
                assert pos_edges + neg_edges == n_bins + 1, (
                    "Expecting at most 1 missing edge. Got "
                    f"{pos_edges + neg_edges}, expected {n_bins + 1} or {n_bins + 2}."
                )

                # Determine which side to add an edge to.
                ideal_pos_ratio = vmax / vrange
                if pos_edges / (pos_edges + neg_edges) >= ideal_pos_ratio:
                    # We have too many positive edges already, so increment the number
                    # of negative edges.
                    neg_edges += 1
                else:
                    pos_edges += 1

            return list(np.linspace(vmin, 0, pos_edges)) + list(
                np.linspace(0, vmax, neg_edges)[1:]
            )

    # Only the log case remains here.

    if min_edge_type not in ("symmetric", "auto", "manual"):
        raise ValueError(f"Unexpected `min_edge_type` {min_edge_type}.")

    if min_edge_type == "manual":
        assert min_edge is not None, "Need valid `min_edge` for 'manual' edge type."

    if min_edge is not None:
        if isinstance(min_edge, (float, int, np.float, np.integer)):
            min_edge = [min_edge, min_edge]
        min_edge = np.abs(np.asarray(min_edge, dtype=np.float64))

    # Handle the positive and negative data ranges separately.

    # Positive and negative data are handled the same way (after taking the absolute
    # value) and the resulting bins are then stitched together if needed.

    if data is not None:
        zero_mask = ~np.isclose(data, 0)
        split_data = (
            data[(data > 0) & zero_mask],
            np.abs(data[(data < 0) & zero_mask]),
        )
    else:
        split_data = (None, None)

    if vmin >= 0:
        split_data = split_data[:1]
    elif vmax <= 0:
        split_data = split_data[1:]

    if min_edge is not None:
        if min_edge_type != "manual":
            raise ValueError(
                "Value for `min_edge` supplied even though `min_edge_type` was set to "
                f"'{min_edge_type}'."
            )

    if min_edge_type in ("symmetric", "auto"):
        # This guarantees that data is not None, meaning that we can use the zero
        # mask.
        if min_edge_type == "symmetric":
            # Use the same minimum edge for both data ranges if needed.
            min_edge_type = "manual"
            min_edge = [np.min(np.abs(data[zero_mask]))] * 2
        else:
            # If edge type is auto, data is needed to infer the bounds.
            if not all(map(np.any, split_data)):
                raise ValueError(
                    f"Insufficient data for `min_edge_type` '{min_edge_type}'."
                )
            # Use the data to infer the minimum edge separately for each data range.
            min_edge = list(map(np.min, split_data))

    # List that holds the output bin edges.
    bin_edges = []

    if vmin >= 0:
        multipliers = (1,)
        limits = [[vmin, vmax]]
        if np.isclose(limits[0][0], 0):
            bin_edges.append(limits[0][0])
            if isinstance(n_bins, (int, np.integer)):
                n_bins -= 1  # Compensate for the additional edge above.
            limits[0][0] = min_edge[0]
    elif vmax <= 0:
        multipliers = (-1,)
        limits = [[np.abs(vmax), np.abs(vmin)]]
        if np.isclose(limits[0][0], 0):
            bin_edges.append(limits[0][0])
            if isinstance(n_bins, (int, np.integer)):
                n_bins -= 1  # Compensate for the additional edge above.
            limits[0][0] = min_edge[1]
    else:
        multipliers = (1, -1)
        limits = [(min_edge[0], vmax), (min_edge[1], np.abs(vmin))]

    if vmin >= 0 or vmax <= 0:
        # Only a single bin number is required.
        bins = [n_bins]
    else:
        if isinstance(n_bins, (int, np.integer)):
            contributions = np.array(
                [np.ptp(np.log10(s_limits)) for s_limits in limits]
            )
            total = np.sum(contributions)
            # To arrive at `n_bins` bins (ie. `n_bins + 1` edges), we need to consider
            # the number of bins for the -ve and +ve part, and the 0-edge in the
            # middle. Concatenating the two ranges (which do not include 0 as opposed
            # to the linear case above) adds 1 bin, while adding the 0-edge adds
            # another bin. Thus, `n_bins - 2` bins need to be provided by the two
            # parts.
            bins = [math.ceil(n_bin) for n_bin in (n_bins - 2) * contributions / total]

            # If we have 1 bin too few (remember that we want `n_bins - 2` bins at
            # this stage.
            if sum(bins) < n_bins - 2:
                ideal_pos_ratio = contributions[0] / total
                if bins[0] / np.sum(bins) >= ideal_pos_ratio:
                    # We have too many positive bins already, so increment the number
                    # of negative bins.
                    bins[0] += 1
                else:
                    bins[1] += 1
        else:
            # Ie. use "auto" bins for both ranges.
            bins = [n_bins, n_bins]

    if n_close_lim == 1:
        assert (
            len(split_data) == 1
        ), "There should be only 1 split dataset if one of the limits is (close to) 0."

    # Now that the ranges have been segregated, handle them each individually and
    # combine the resulting bin edges.

    # Get the relevant limits for the data in both the positive and negative case.
    # Do this by checking the sign and magnitude of the predefined limits, and
    # using max/min of the selected data if necessary.
    for multiplier, s_limits, s_bins in zip(multipliers, limits, bins):
        bin_edges.extend(multiplier * np.asarray(_get_log_bin_edges(*s_limits, s_bins)))

    if len(split_data) == 2:
        bin_edges.append(0)

    return sorted(float(edge) for edge in bin_edges)


def cube_plotting(
    cube,
    log=False,
    rasterized=True,
    coastline_kwargs={},
    dummy_lat_lims=(-90, 90),
    dummy_lon_lims=(-180, 180),
    vmin_vmax_percentiles=(0, 100),
    projection=None,
    animation_output=False,
    ax=None,
    mesh=None,
    new_colorbar=True,
    title_text=None,
    transform_vmin_vmax=False,
    nbins=10,
    log_auto_bins=True,
    boundaries=None,
    min_edge=None,
    extend=None,
    average_first_coord=True,
    select_valid=False,
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
        projection: A projection as defined in `cartopy.crs`. If None (default),
            ccrs.Robinson() will be used, where the central longitude will be defined
            as the average of the cube longitudes.
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
        boundaries (iterable or None): If None, bin boundaries will be computed
            automatically. Supersedes all other options relating to boundary creation,
            like `log' or `vmin'.
        min_edge (float or None): Minimum log bin exponent. See `get_bin_edges`.
        extend (None or str): If None, determined based on given vmin/vmax. If vmin is
            given, for example, `extend='min'`. If vmax is given, for example,
            `extend='max'`. If both vmin & vmax are given, `extend='both'`. This value
            can be set manually to one of the aforementioned options.
        average_first_coord (bool): Average out first coordinate if there are 3
            dimensions.
        select_valid (bool): If True, select central contiguous unmasked subset of
            data.
        possible kwargs:
            title: str or None of False. If None or False, no title will be plotted.
            cmap: Example: 'viridis', 'Reds', 'Reds_r', etc... Can also be a
                `matplotlib.colors.Colormap` instance.
            vmin: Minimum value for colorbar.
            vmax: Maximum value for colorbar.
            cbar_tick_size: Colorbar tick param text size.
            cbar_label_size:
        possible colorbar kwargs:
            label:
            orientation:
            fraction:
            pad:
            shrink:
            aspect:
            anchor:
            panchor:
            format:

    """
    if not isinstance(cube, iris.cube.Cube):
        cube = dummy_lat_lon_cube(
            cube, lat_lims=dummy_lat_lims, lon_lims=dummy_lon_lims
        )
    if hasattr(cube.data, "mask"):
        if np.all(cube.data.mask):
            raise MaskedDataError("All data is masked.")

    if not hasattr(cube.data, "mask"):
        cube.data = np.ma.MaskedArray(cube.data, mask=False)

    if select_valid and not np.all(~cube.data.mask):
        cube, tr_longitudes = select_valid_subset(
            cube, longitudes=cube.coord("longitude").points
        )
        central_longitude = np.mean(tr_longitudes)
    else:
        longitudes = cube.coord("longitude").points
        if in_360_longitude_system(longitudes):
            # Translate longitudes to centre the map on Africa when all longitudes are
            # present and a choice has to be made.
            logger.debug("Translating longitudes from [0, 360] to [-180, 180].")
            central_longitude = np.mean(
                translate_longitude_system(longitudes, return_indices=False)
            )
        else:
            central_longitude = np.mean(longitudes)

    title = kwargs.get("title")
    if title is None:
        # Construct a default title.
        title_list = [cube.name()]
        try:
            time_coord = cube.coord("time")
            min_time = (
                time_coord.cell(0).bound[0]
                if time_coord.cell(0).bound is not None
                else time_coord.cell(0).point
            )
            max_time = (
                time_coord.cell(-1).bound[1]
                if time_coord.cell(-1).bound is not None
                else time_coord.cell(-1).point
            )
            if min_time == max_time:
                title_list.append(f"{min_time}")
            else:
                title_list.append(f"{min_time} - {max_time}")
        except iris.exceptions.CoordinateNotFoundError:
            # No time coordinate was found, so we cannot use it in the title.
            pass

        title = "\n".join(title_list)

    if projection is None:
        logger.debug(f"Central longitude ({title}): {central_longitude:0.2f}")
        projection = ccrs.Robinson(central_longitude=central_longitude)

    if average_first_coord and len(cube.shape) == 3:
        cube = cube.collapsed(cube.coords()[0], iris.analysis.MEAN)

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

        if boundaries is None:
            boundaries = get_bin_edges(
                cube.data,
                vmin,
                vmax,
                "auto" if log and log_auto_bins else nbins,
                log,
                "symmetric" if min_edge is None else "manual",
                min_edge=min_edge,
            )

        if extend is None:
            if vmin is None and vmax is None:
                extend = "neither"
            if vmin is not None and vmax is None:
                extend = "min"
            if vmax is not None and vmin is None:
                extend = "max"
            if vmin is not None and vmax is not None:
                extend = "both"

        raw_cmap = kwargs.get("cmap", "viridis")

        if isinstance(raw_cmap, str):
            if extend == "neither":
                n_colors = len(boundaries) - 1
            elif extend == "min":
                n_colors = len(boundaries)
            elif extend == "max":
                n_colors = len(boundaries)
            elif extend == "both":
                n_colors = len(boundaries) + 1
            else:
                raise ValueError(f"Unknown value for `extend` {repr(extend)}.")

            # Allow manual flipping of colormap.
            cmap_sample_lims = [0, 1]
            try:
                orig_cmap = plt.get_cmap(raw_cmap)
            except ValueError:
                logger.debug(f"Exception while trying to access cmap '{raw_cmap}'.")
                if isinstance(raw_cmap, str) and "_r" in raw_cmap:
                    # Try to reverse the colormap manually, in case a reversed colormap
                    # was requested using the '_r' suffix, but this is not available.
                    raw_cmap = raw_cmap[:-2]
                    orig_cmap = plt.get_cmap(raw_cmap)

                    # Flip limits to achieve reversal effect.
                    cmap_sample_lims = [1, 0]
                    logger.debug(f"Manually reversing cmap '{raw_cmap}'.")
                else:
                    raise

            if n_colors > orig_cmap.N:
                logger.warning(
                    f"Expected at most {orig_cmap.N} " f"colors, but got {n_colors}."
                )
                if n_colors <= 20:
                    orig_cmap = plt.get_cmap("tab20")
                else:
                    orig_cmap = plt.get_cmap("viridis")
                logger.warning(f"Reverting colormap to {orig_cmap.name}.")

            logger.debug(f"Boundaries:{boundaries}.")
            cmap, norm = from_levels_and_colors(
                boundaries,
                orig_cmap(np.linspace(*cmap_sample_lims, n_colors)),
                extend=extend,
            )
        else:
            cmap = raw_cmap
            norm = mpl.colors.Normalize(
                vmin=np.min(boundaries), vmax=np.max(boundaries)
            )

        # This forces data points 'exactly' (very close) to the upper limit of the last
        # bin to be recognised as part of that bin, as opposed to out of bounds.
        if extend == "neither":
            norm.clip = True

        mesh = ax.pcolormesh(
            gridlons,
            gridlats,
            cube.data,
            # cmap=kwargs.get("cmap", "viridis"),
            cmap=cmap,
            norm=norm,
            # NOTE: This transform may differ from the projection argument, because it
            # represents the coordinate system of the input data, as opposed to the
            # coordinate system of the plot.
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
        "cax": kwargs.get("cax"),
        "ax": ax,
    }
    # TODO: https://matplotlib.org/3.1.0/gallery/axes_grid1/simple_colorbar.html
    # TODO: Use this to attach the colorbar to the axes, not the figure. Errors were
    # encountered due to cartopy geoaxes.
    if new_colorbar:
        cbar = fig.colorbar(mesh, **colorbar_kwargs)
        if "cbar_tick_size" in kwargs:
            cbar.ax.tick_params(labelsize=kwargs["cbar_tick_size"])
        if "cbar_label_size" in kwargs:
            cbar.set_label(
                label=colorbar_kwargs["label"], size=kwargs["cbar_label_size"]
            )
    if title:
        if isinstance(title, mpl.text.Text):
            title_text.set_text(title)
        else:
            title_text = fig.suptitle(title)

    if animation_output:
        return fig, ax, mesh, title_text
    return fig


def get_pdp_data(
    model,
    X,
    features,
    grid_resolution=100,
    percentiles=(0.0, 1.0),
    coverage=1,
    random_state=None,
):
    """Get data for 1 dimensional partial dependence plots.

    Args:
        model: A fitted model with a 'predict' method.
        X (pandas.DataFrame): Dataframe containing the data over which the
            partial dependence plots are generated.
        features (list): A list of int or string with which to select
            columns from X. The type must match the column labels.
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

    Returns:
        4-tuple of pandas.DataFrame: Quantiles, average predictions, minimum
            predictions and maximum predictions for each feature.

    """
    features = list(features)

    quantiles = X[features].quantile(percentiles)
    quantile_data = pd.DataFrame(
        np.linspace(quantiles.iloc[0], quantiles.iloc[1], grid_resolution),
        columns=features,
    )

    datasets = []
    min_datasets = []
    max_datasets = []

    if not np.isclose(coverage, 1):
        logger.debug("Selecting subset of data with coverage:{:}".format(coverage))
        np.random.seed(random_state)

        # Select a random subset of the data.
        permuted_indices = np.random.permutation(np.arange(X.shape[0]))
        permuted_indices = permuted_indices[: int(coverage * X.shape[0])]
        X_selected = X.iloc[permuted_indices]
    else:
        logger.debug("Selecting all data.")
        X_selected = X

    for feature in tqdm(features):
        series = quantile_data[feature]

        # for each possible value of the selected feature, average over all
        # possible combinations of the other features
        averaged_predictions = []
        min_predictions = []
        max_predictions = []
        calc_X = X_selected.copy()
        for value in series:
            calc_X[feature] = value
            prediction = model.predict(calc_X)

            averaged_predictions.append(np.mean(prediction))
            min_predictions.append(np.min(prediction))
            max_predictions.append(np.max(prediction))

        datasets.append(np.array(averaged_predictions).reshape(-1, 1))
        min_datasets.append(np.array(min_predictions).reshape(-1, 1))
        max_datasets.append(np.array(max_predictions).reshape(-1, 1))

    avg_results = pd.DataFrame(np.hstack(datasets), columns=features)
    min_results = pd.DataFrame(np.hstack(min_datasets), columns=features)
    max_results = pd.DataFrame(np.hstack(max_datasets), columns=features)

    return quantile_data, avg_results, min_results, max_results


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
    predicted_name="Burned Area",
    norm_y_ticks=False,
    single_plots=False,
    keep_y_ticks=False,
    log_x_scale=None,
    X_train=None,
    plot_range=True,
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
        single_plots (bool): If True, plot `n_features` plots instead of combining
            them.
        keep_y_ticks (bool): If True, do not turn off y-axis ticks.
        log_x_scale (iterable): If given, features matching one of the supplied
            strings will be logged (symlog) when plotting.
        X_train (pandas.DataFrame): Similar to `X`. Is plotted using the same
            transformations used for `X`, but using dotted lines. Can be used eg. for
            comparison between train and test sets.
        plot_range (bool): If True, plot the minimum and maximum predictions.

    Returns:
        fig, ax or ((fig1, ax1), (fig2, ax2), ...): Figure and axes holding the pdp
            (see `single_plots`).

    """
    quantile_data, results, min_results, max_results = get_pdp_data(
        model, X, features, grid_resolution, percentiles, coverage, random_state
    )
    if X_train is not None:
        (
            train_quantile_data,
            train_results,
            train_min_results,
            train_max_results,
        ) = get_pdp_data(
            model,
            X_train,
            features,
            grid_resolution,
            percentiles,
            coverage,
            random_state,
        )

    def pdp_axis_plot(ax, feature):
        ax.plot(quantile_data[feature], results[feature], c="C0", zorder=3)
        if plot_range:
            ax.fill_between(
                quantile_data[feature],
                min_results[feature],
                max_results[feature],
                facecolor="C0",
                zorder=1,
                alpha=0.3,
            )

        if X_train is not None:
            ax.plot(
                train_quantile_data[feature],
                train_results[feature],
                linestyle="--",
                c="C1",
                zorder=2,
            )
            if plot_range:
                ax.fill_between(
                    train_quantile_data[feature],
                    train_min_results[feature],
                    train_max_results[feature],
                    facecolor="C1",
                    zorder=0,
                    alpha=0.15,
                )

        ax.set_xlabel(feature)

        if log_x_scale is not None:
            if any(name in feature for name in log_x_scale):
                ax.set_xscale("symlog")

    if single_plots:
        figs_axes = []
        for feature in features:
            fig, ax = plt.subplots()
            figs_axes.append((fig, ax))
            pdp_axis_plot(ax, feature)
            ax.set_ylabel(predicted_name)

        if not keep_y_ticks:
            for fig, ax in figs_axes:
                ax.axes.get_yaxis().set_ticks([])
        return figs_axes
    else:
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
            pdp_axis_plot(ax, feature)

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

        if not keep_y_ticks:
            for ax in axes:
                ax.axes.get_yaxis().set_ticks([])
        plt.tight_layout()

        return fig, axes


def sample_pdp():
    import string

    np.random.seed(1)

    class A:
        def predict(self, X):
            # Prevent modification from persisting.
            X = X.copy()
            X.iloc[:, 0] *= -(0.4 + np.mean(X.iloc[:, 0]) + X.iloc[0, 1])

            return np.sum(X, axis=1)

    test_data = np.zeros((2, 8)) + np.array([[1], [2]])

    test_data += np.random.random(test_data.shape) - 0.5
    test_data2 = test_data + 0.2 * (np.random.random(test_data.shape) - 0.5)

    model = A()
    X = pd.DataFrame(
        test_data, columns=list(string.ascii_lowercase)[: test_data.shape[1]]
    )
    X2 = pd.DataFrame(
        test_data2, columns=list(string.ascii_lowercase)[: test_data.shape[1]]
    )
    features = list(string.ascii_lowercase[: test_data.shape[1]])

    fig, axes = partial_dependence_plot(
        model,
        X,
        features,
        grid_resolution=10,
        norm_y_ticks=False,
        percentiles=(0, 1),
        X_train=X2,
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


def sample_region_plotting():
    np.random.seed(1)
    shape = (200, 400)
    cube = dummy_lat_lon_cube(np.random.random(shape) ** 3 + 0.05, units="1")
    cube.data = np.ma.MaskedArray(
        cube.data, mask=np.zeros_like(cube.data, dtype=np.bool_)
    )
    cube.data.mask[:50] = True
    cube.data.mask[150:] = True
    cube.data.mask[:, :50] = True
    cube.data.mask[:, 150:] = True
    cube_plotting(
        cube, log=True, title="Testing", orientation="horizontal", select_valid=True
    )
    plt.show()


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)

    FigureSaver.directory = os.path.join(
        os.path.expanduser("~"), "tmp", "plotting_test"
    )
    os.makedirs(FigureSaver.directory, exist_ok=True)
    FigureSaver.debug = True

    with FigureSaver("pdp_test"):
        sample_pdp()
