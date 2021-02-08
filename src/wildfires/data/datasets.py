# -*- coding: utf-8 -*-
"""Module that simplifies use of various datasets.

TODO:
    Figure out how to handle the fact that saving a cube for certain kinds of
    longitude coordinates adds the attribute `circular=True`.

    Since longitudes registered in the [-180, 180] system fail to register as circular
    using `iris.util._is_circular` but the same coordinates + 180° do register
    correctly (if they actually cover all of [0, 360], as expected), does regridding
    using longitudes in [-180, 180] actually consider the coordinate's circular nature
    correctly?
    See also NOTE in `lat_lon_dimcoords`.

    Before caching, regrid() always needs to be called since this also carries out
    crucial coordinate attribute regularisation, which is essential for consistent
    caching behaviour! Make this intrinsic to __init__?

"""
import concurrent.futures
import glob
import logging
import operator
import os
import re
import shutil
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import reduce, wraps

import cf_units
import dask.array as da
import h5py
import iris
import iris.coord_categorisation
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage
from dateutil.relativedelta import relativedelta
from era5analysis import MonthlyMeanMinMaxDTRWorker, retrieval_processing, retrieve
from git import InvalidGitRepositoryError, Repo
from iris.time import PartialDateTime
from joblib import Memory, Parallel, delayed
from numpy.testing import assert_allclose
from pyhdf.SD import SD, SDC
from scipy.optimize import minimize
from tqdm import tqdm

from .. import __version__
from ..chunked_regrid import spatial_chunked_regrid
from ..joblib.caching import CodeObj, wrap_decorator
from ..joblib.iris_backend import register_backend
from ..qstat import get_ncpus
from ..utils import (
    box_mask,
    ensure_datetime,
    get_centres,
    get_land_mask,
    in_360_longitude_system,
    match_shape,
    reorder_cube_coord,
    strip_multiline,
    translate_longitude_system,
)
from .landcover import conversion as lc_to_pft_map
from .landcover import convert_to_pfts

__all__ = (
    "CommitMatchError",
    "DATA_DIR",
    "Dataset",
    "DatasetNotFoundError",
    "Error",
    "IGNORED_DATASETS",
    "MM_PER_HR_THRES",
    "M_PER_HR_THRES",
    "MonthlyDataset",
    "NonUniformCoordError",
    "ObservedAreaError",
    "VariableNotFoundError",
    "cube_contains_coords",
    "data_is_available",
    "data_map_plot",
    "dataset_cache",
    "dataset_preprocessing",
    "dataset_times",
    "dummy_lat_lon_cube",
    "fill_cube",
    "fill_dataset",
    "get_climatology",
    "get_dataset_climatology_cubes",
    "get_dataset_mean_cubes",
    "get_dataset_monthly_cubes",
    "get_mean",
    "get_memory",
    "get_monthly",
    "get_monthly_mean_climatology",
    "homogenise_cube_attributes",
    "homogenise_cube_mask",
    "homogenise_time_coordinate",
    "join_adjacent_intervals",
    "lat_lon_dimcoords",
    "lat_lon_match",
    "load_cubes",
    "monthly_constraint",
    "regions_GFED",
    "regrid",
    "regrid_dataset",
)


IGNORED_DATASETS = [
    "AvitabileAGB",
    "CRU",
    "ESA_CCI_Fire",
    "ESA_CCI_Landcover",
    "ESA_CCI_Soilmoisture",
    "ESA_CCI_Soilmoisture_Daily",
    "GPW_v4_pop_dens",
    "LIS_OTD_lightning_time_series",
    "Simard_canopyheight",
    "Thurner_AGB",
]


logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.expanduser("~"), "FIREDATA")


repo_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
try:
    repo = Repo(repo_dir)
except InvalidGitRepositoryError:
    repo = None

# Above this mm/h threshold, a day is a 'wet day'.
# 0.1 mm per day, from Harris et al. 2014, as used in Forkel et al. 2018.
MM_PER_HR_THRES = 0.1 / 24
M_PER_HR_THRES = MM_PER_HR_THRES / 1000


def data_is_available():
    """Check if DATA_DIR exists.

    Returns:
        bool: True if the data directory exists.

    """
    return os.path.exists(DATA_DIR)


def get_memory(cache_dir="", **kwargs):
    """Get a joblib Memory object used to cache function results.

    Args:
        cache_dir (str or None): Joblib cache directory name within
            `wildfires.data.DATA_DIR`. If None, no caching will be done.
        **kwargs: Extra arguments passed to `joblib.Memory()`.

    Returns:
        joblib.memory.Memory: Joblib Memory object.

    """
    return Memory(
        location=os.path.join(DATA_DIR, "joblib_cache", cache_dir)
        if cache_dir is not None and data_is_available()
        else None,
        **kwargs,
    )


register_backend()
iris_memory = get_memory("datasets", backend="iris", verbose=0)


class Error(Exception):
    """Base class for exceptions in the datasets module."""


class ObservedAreaError(Error):
    """Raised when a Dataset does not satisfy observed area calculation requirements."""


class NonUniformCoordError(Error):
    """Raised when a coordinate is neither monotonically increasing or decreasing."""


class CommitMatchError(Error):
    """Raised when commit hashes of loaded cubes do not match."""


class VariableNotFoundError(ValueError, Error):
    """Raised when a requested variable could not be found."""


class DatasetNotFoundError(ValueError, Error):
    """Raised when a requested dataset could not be found."""


class UnexpectedCoordinateError(ValueError, Error):
    """Raised when an unexpected coordinate is encountered."""


def fill_cube(cube, mask):
    """Process cube in-place by filling gaps using NN interpolation and also filtering.

    The idea is to respect the masking of one variable (usually burned area) alone.
    For all others, replace masked data using nearest-neighbour interpolation.
    Thereafter, apply the aggregated mask `mask`, so that eg. only data over land and
    within the latitude limits is considered. Latitude limits might be due to
    limitations of GSMaP precipitation data, as well as limitations of the lightning
    LIS/OTD dataset, for example.

    Args:
        cube (iris.cube.Cube): Cube to be filled.
        mask (numpy.ndarray): Boolean mask typically composed of the land mask,
            latitude mask and burned area mask like:
                `mask=land_mask | lat_mask | burned_area_mask`.
            This controls which data points remain after the processing, while the
            mask internal to the cube's data controls where interpolation will take
            place.

    """
    assert isinstance(cube.data, np.ma.core.MaskedArray) and isinstance(
        cube.data.mask, np.ndarray
    ), "Cube needs to have a full (non-sparse) data mask."

    # Here, data gaps are filled, so that the maximum possible area of data
    # (limited by where burned area data is available) is used for the analysis.
    # Choose to fill the gaps using nearest-neighbour interpolation. To do this,
    # define a mask which will tell the algorithm where to replace data.

    logger.info("Filling: '{}'.".format(cube.name()))

    # Interpolate data where (and if) it is masked.
    fill_mask = cube.data.mask
    if np.sum(fill_mask[~mask]):
        orig_data = cube.data.data.copy()
        logger.info(
            "Filling {:} elements ({:} after final masking).".format(
                np.sum(fill_mask), np.sum(fill_mask[~mask])
            )
        )
        filled_data = cube.data.data[
            tuple(
                scipy.ndimage.distance_transform_edt(
                    fill_mask, return_distances=False, return_indices=True
                )
            )
        ]
        assert np.all(np.isclose(cube.data.data[~fill_mask], orig_data[~fill_mask]))

        selected_unfilled_data = orig_data[~mask]
        selected_filled_data = filled_data[~mask]

        logger.info(
            "Min {:0.1e}/{:0.1e}, max {:0.1e}/{:0.1e} before/after "
            "filling (for relevant regions)".format(
                np.min(selected_unfilled_data),
                np.min(selected_filled_data),
                np.max(selected_unfilled_data),
                np.max(selected_filled_data),
            )
        )
    else:
        # Prevent overwriting with previous loop's filled data if there is nothing
        # to fill.
        filled_data = cube.data.data

    # Always apply global combined mask.
    cube.data = np.ma.MaskedArray(filled_data, mask=mask)

    # Check that there aren't any inf's or nan's in the data.
    assert not np.any(np.isinf(cube.data.data[~cube.data.mask]))
    assert not np.any(np.isnan(cube.data.data[~cube.data.mask]))

    return cube


def cube_contains_coords(cube, *coords):
    """Check whether the given cube contains all the requested coordinates."""
    for coord in coords:
        if not cube.coords(coord):
            return False
    return True


def homogenise_time_coordinate(cubes):
    """Ensure all given cubes have the same time coordinate.

    Args:
        cubes (iris.cube.CubeList): List of cubes to process.

    Returns:
        iris.cube.CubeList: List of processed cubes.

    Raises:
        ValueError: If any cube does not have a "time" coordinate.
        ValueError: If `cubes` have differing calendars.

    """
    # Check that all cubes have a "time" coordinate.
    for cube in cubes:
        if not cube.coords("time"):
            raise ValueError(f"Cube '{cube}' did not have a 'time' coordinate.")

    # Check that all cubes have the same calendar.
    calendars = {cube.coord("time").units.calendar for cube in cubes}
    if len(calendars) > 1:
        raise ValueError(f"Cubes had differing calendars: '{calendars}'.")

    # Define the common calendar to use.
    calendar = list(calendars)[0]
    ref_time = datetime(1970, 1, 1)
    ref_unit = cf_units.Unit(
        f"days since {ref_time.strftime('%Y-%m-%d %H:%M:%S')}", calendar=calendar
    )
    # Convert all times (and bounds).
    for cube in cubes:
        cube.coord("time").convert_units(ref_unit)

    return cubes


def homogenise_cube_attributes(cubes, adjust_time=False):
    """Ensure all given cubes have compatible attributes in-place.

    Args:
        cubes (iris.cube.CubeList): List of cubes to process.
        adjust_time (bool): If True, modify the temporal coordinates (with name
            "time") to have the same units.

    Returns:
        iris.cube.CubeList: List of processed cubes.

    Raises:
        ValueError: If `adjust_time` is True and any cube does not have a "time"
            coordinate.
        ValueError: If `adjust_time` is True and `cubes` have differing calendars.

    """
    attribute_list = [cube.attributes for cube in cubes]
    common_values = attribute_list[0].copy()

    for attributes in attribute_list[1:]:
        shared_keys = set(common_values).intersection(set(attributes))
        common_values = dict(
            (key, common_values[key])
            for key in shared_keys
            if common_values[key] == attributes[key]
        )

    for cube in cubes:
        cube.attributes = common_values

    if adjust_time:
        homogenise_time_coordinate(cubes)

    return cubes


@wrap_decorator
def dataset_cache(func, iris_memory=iris_memory):
    # NOTE: There is a known bug preventing joblib from pickling numpy MaskedArray!
    # NOTE: https://github.com/joblib/joblib/issues/573
    # NOTE: We will avoid this bug by replacing Dataset instances (which may hold
    # NOTE: references to masked arrays) with their (shallow) immutable string
    # NOTE: representations.
    """Circumvent bug preventing joblib from pickling numpy MaskedArray instances.

    This applies to MaskedArray in the input arguments only.

    Do this by giving joblib a different version of the input arguments to cache,
    while still passing the normal arguments to the decorated function.

    Note:
        `dataset_function` argument in `func` must be a keyword argument.

    """

    @wraps(func)
    def accepts_dataset(*orig_args, **orig_kwargs):
        """Function that is visible to the outside."""
        if not isinstance(orig_args[0], Dataset) or any(
            isinstance(arg, Dataset)
            for arg in list(orig_args[1:]) + list(orig_kwargs.values())
        ):
            raise TypeError(
                "The first positional argument, and only the first argument "
                f"should be a `Dataset` instance, got '{type(orig_args[0])}' "
                "as the first argument."
            )
        dataset = orig_args[0]
        string_representation = dataset._shallow

        # Ignore instances with a __call__ method here which also wouldn't necessarily
        # have a __name__ attribute that could be used for sorting!
        functions = [func]
        func_code = tuple(CodeObj(f.__code__).hashable() for f in functions)

        @iris_memory.cache(ignore=["original_dataset"])
        def accepts_string_dataset(
            func_code, string_representation, original_dataset, *args, **kwargs
        ):
            # NOTE: The reason why this works is that the combination of
            # [original_selection] + args here is fed the original `orig_args`
            # iterable. In effect, the `original_selection` argument absorbs one of
            # the elements of `orig_args` in the function call to
            # `takes_split_selection`, so it is not passed in twice. The `*args`
            # parameter above absorbs the rest. This explicit listing of
            # `original_selection` is necessary, as we need to explicitly ignore
            # `original_selection`, which is the whole point of this decorator.
            out = func(original_dataset, *args, **kwargs)
            return out

        return accepts_string_dataset(
            func_code, string_representation, *orig_args, **orig_kwargs
        )

    return accepts_dataset


def homogenise_cube_mask(cube):
    """Ensure cube.data is a masked array with a full mask (in-place).

    Note:
        This function realises lazy data.

    """
    array = cube.data
    if isinstance(array, np.ma.core.MaskedArray):
        if isinstance(array.mask, np.ndarray):
            return cube
        else:
            if array.mask:
                raise ValueError(
                    "The only mask entry is True, meaning all entries are masked!"
                )
    cube.data = np.ma.MaskedArray(array, mask=np.zeros_like(array, dtype=np.bool_))
    return cube


def dataset_preprocessing(dataset, min_time, max_time):
    """Process `Dataset` `dataset` in-place to enforce uniformity."""
    # This step does take on the order of seconds, but re-caching here would be a
    # waste of space.
    # Limit the amount of data that has to be processed.
    dataset.limit_months(min_time, max_time)

    # Regrid cubes to the same lat-lon grid.
    # TODO: change lat and lon limits and also the number of points!!
    # Always work in 0.25 degree steps? From the same starting point?
    dataset.regrid()


def get_monthly_mean_climatology(dataset, min_time, max_time, *args, **kwargs):
    """Modifies `dataset` in-place, also generating temporal averages.

    TODO:
        Currently `dataset` is modified by regridding, temporal selection and
        `get_monthly_data`. Should this be circumvented by copying it?

    Returns:
        tuple of `Dataset`

    """
    dataset_preprocessing(dataset, min_time, max_time)

    # Generate overall temporal mean. Do this before monthly data is created for all
    # datasets, since this will only increase the computational burden and skew the
    # mean towards newly synthesised months (created using `get_monthly_data`,
    # for static, yearly, or climatological datasets).
    mean_dataset = dataset.get_mean_dataset()

    climatology_dataset = dataset.get_climatology_dataset(min_time, max_time)

    # Get monthly data over the chosen interval for all dataset.
    # TODO: Inplace argument for get_monthly_data methods?
    monthly_dataset = dataset.get_monthly_dataset(min_time, max_time)

    # TODO: See note in `get_mean_climatology_monthly_dataset`.
    # mean_dataset, climatology_dataset, monthly_dataset = monthly_dataset.get_mean_climatology_monthly_dataset(
    #     min_time, max_time
    # )

    return monthly_dataset, mean_dataset, climatology_dataset


def get_monthly(dataset, min_time, max_time):
    dataset_preprocessing(dataset, min_time, max_time)
    return (dataset.get_monthly_dataset(min_time, max_time),)


def get_climatology(dataset, min_time, max_time):
    dataset_preprocessing(dataset, min_time, max_time)
    return (dataset.get_climatology_dataset(min_time, max_time),)


def get_mean(dataset, min_time, max_time):
    dataset_preprocessing(dataset, min_time, max_time)
    return (dataset.get_mean_dataset(),)


@dataset_cache
def fill_dataset(dataset, mask):
    """Perform processing on all cubes."""
    logger.debug(f"Filling '{dataset}' with {len(dataset)} variable(s).")
    return iris.cube.CubeList([fill_cube(cube, mask) for cube in dataset])


@dataset_cache
def get_dataset_mean_cubes(dataset):
    """Return mean cubes."""
    logger.debug(f"Calculating mean for '{dataset}' with {len(dataset)} variable(s).")
    # TODO: Should we copy the cube if it does not require averaging?
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*Collapsing a non-contiguous coordinate.*"
        )
        mean_cubes = iris.cube.CubeList(
            cube.collapsed("time", iris.analysis.MEAN) if cube.coords("time") else cube
            for cube in dataset
        )
    for cube in mean_cubes:
        # This function has the wanted side-effect of realising the data.
        # Without this, calculations of things like the total temporal mean are
        # delayed until the cube is needed, which we do not want here as we are
        # caching the results.
        homogenise_cube_mask(cube)

    # This return value will be cached by writing it to disk as NetCDF files.
    return mean_cubes


@dataset_cache
def get_dataset_monthly_cubes(dataset, start, end):
    """Return monthly cubes between two dates."""
    logger.debug(
        f"Calculating monthly cubes for '{dataset}' with {len(dataset)} variable(s)."
    )
    monthly_cubes = dataset.get_monthly_data(start, end)
    for cube in monthly_cubes:
        # This function has the wanted side-effect of realising the data.
        # Without this, calculations of things like the total temporal mean are
        # delayed until the cube is needed, which we do not want here as we are
        # caching the results.
        homogenise_cube_mask(cube)

    # This return value will be cached by writing it to disk as NetCDF files.
    return monthly_cubes


@dataset_cache
def get_dataset_climatology_cubes(dataset, start, end):
    logger.debug(
        f"Calculating climatology for '{dataset}' with {len(dataset)} variable(s)."
    )
    # NOTE: Calling get_dataset_monthly_cubes using the slices is important, as this is
    # how it is called originally, and therefore how it is represented in the cache!!
    # TODO: Make this behaviour more transparent - perhaps embed this in the
    # dataset_cache decorator??

    # TODO: Instead of calling get_dataset_monthly_cubes (which fetches the cache,
    # TODO: requiring file loading, which is slow) access a cached in-memory version of the
    # TODO: monthly cubes somehow!
    # TODO: Use pre-computed results if they are available and passed in via an
    # TODO: optional parameter (eg. `optional_dataset`) (NOT the `dataset` parameter, which
    # TODO: should still point to the original, unadulterated dataset without monthly
    # TODO: processing, as that would be the reference point for climatology retrieval from
    # TODO: the cache).
    # if (
    #     optional_dataset.min_time == PartialDateTime(start.year, start.month)
    #     and optional_dataset.max_time == PartialDateTime(end.year, end.month)
    #     and optional_dataset.frequency == "monthly"
    # ):
    #     monthly_cubes = dataset.cubes
    # else:

    monthly_cubes = iris.cube.CubeList(
        get_dataset_monthly_cubes(dataset[cube_slice], start, end)[0]
        for cube_slice in dataset.single_cube_slices()
    )

    climatology_cubes = iris.cube.CubeList()
    # Calculate monthly climatology.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*Collapsing a non-contiguous coordinate.*"
        )
        for cube in monthly_cubes:
            if not cube.coords("month_number"):
                iris.coord_categorisation.add_month_number(cube, "time")
            climatology_cubes.append(
                cube.aggregated_by("month_number", iris.analysis.MEAN)
            )
    for i in range(len(climatology_cubes)):
        # This function has the wanted side-effect of realising the data.
        # Without this, calculations of things like the total temporal mean are
        # delayed until the cube is needed, which we do not want here as we are
        # caching the results.
        homogenise_cube_mask(climatology_cubes[i])

        sort_indices = np.argsort(climatology_cubes[i].coord("month_number").points)
        if not np.all(sort_indices == np.arange(len(sort_indices))):
            # Reorder cubes to let month numbers increase monotonically if needed.
            climatology_cubes[i] = reorder_cube_coord(
                climatology_cubes[i], sort_indices, name="month_number", promote=False
            )
        # Promote the month_number coordinate to being the leading coordinate.
        iris.util.promote_aux_coord_to_dim_coord(climatology_cubes[i], "month_number")

    # This return value will be cached by writing it to disk as NetCDF files.
    return climatology_cubes


def join_adjacent_intervals(intervals):
    """Join adjacent or overlapping intervals into contiguous intervals.

    Args:
        intervals (list of 2-element iterables): A list of iterables with 2
            elements where each such iterable (eg. the tuple (start, end))
            defines the start and end of the interval.

    Returns:
        list of list: Contiguous intervals.

    Examples:
        >>> join_adjacent_intervals([[1, 2], [2, 3], [-1, 1]])
        [[-1, 3]]
        >>> from datetime import datetime
        >>> contiguous = join_adjacent_intervals([
        ...     (datetime(2000, 1, 1), datetime(2000, 2, 1)),
        ...     (datetime(1999, 1, 1), datetime(2000, 1, 1)),
        ...     (datetime(1995, 1, 1), datetime(1995, 2, 1))
        ...     ])
        >>> contiguous == [
        ...     [datetime(1995, 1, 1), datetime(1995, 2, 1)],
        ...     [datetime(1999, 1, 1), datetime(2000, 2, 1)],
        ...     ]
        True
        >>> overlapping_contiguous = join_adjacent_intervals([
        ...     (datetime(1999, 1, 1), datetime(2000, 2, 1)),
        ...     (datetime(2000, 1, 1), datetime(2000, 2, 1)),
        ...     (datetime(1995, 1, 1), datetime(1995, 3, 1)),
        ...     (datetime(1995, 2, 1), datetime(1995, 4, 1)),
        ...     (datetime(1995, 4, 1), datetime(1995, 5, 1)),
        ...     ])
        >>> overlapping_contiguous == [
        ...     [datetime(1995, 1, 1), datetime(1995, 5, 1)],
        ...     [datetime(1999, 1, 1), datetime(2000, 2, 1)],
        ...     ]
        True
        >>> join_adjacent_intervals([]) == []
        True

    """
    if not intervals:
        return []
    intervals = list(map(list, intervals))
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    contiguous_intervals = [sorted_intervals.pop(0)]
    while sorted_intervals:
        if sorted_intervals[0][0] <= contiguous_intervals[-1][1]:
            contiguous_intervals[-1][1] = max(
                [sorted_intervals.pop(0)[1], contiguous_intervals[-1][1]]
            )
        else:
            contiguous_intervals.append(sorted_intervals.pop(0))
    return contiguous_intervals


def dummy_lat_lon_cube(
    data, lat_lims=(-90, 90), lon_lims=(-180, 180), monthly=False, **kwargs
):
    """Construct a cube from data given certain assumptions and optional arguments.

    Args:
        lat_lims (2-tuple):
        lon_lims (2-tuple):
        monthly (bool): If True, the dummy temporal coordinate will have monthly
            instead of daily increments. Note that this is only used if the
            `dim_coords_and_dims` keyword argument is not given.
        kwargs:
            Of note are:
                - dim_coords_and_dims: If supplied, will be use to initialise
                      coordinates instead of `lat_lims`, `lon_lims` and a simple
                      numerical time coordinate.

    """
    n_dims = len(data.shape)
    assert n_dims in {2, 3}
    new_latitudes = get_centres(np.linspace(*lat_lims, data.shape[0 + n_dims % 2] + 1))
    new_longitudes = get_centres(np.linspace(*lon_lims, data.shape[1 + n_dims % 2] + 1))
    new_lat_coord = iris.coords.DimCoord(
        new_latitudes, standard_name="latitude", units="degrees"
    )
    new_lon_coord = iris.coords.DimCoord(
        new_longitudes,
        standard_name="longitude",
        units="degrees",
        circular=True,
    )

    if n_dims == 2:
        grid_coords = [(new_lat_coord, 0), (new_lon_coord, 1)]
    else:
        # Define a time coordinate as well.
        units = cf_units.Unit("days since 1970-01-01", calendar="gregorian")
        if monthly:
            num_dates = units.date2num(
                [
                    datetime(1970, 1, 1) + relativedelta(months=months)
                    for months in range(data.shape[0])
                ]
            )
        else:
            num_dates = list(range(data.shape[0]))
        time_coord = iris.coords.DimCoord(
            num_dates, standard_name="time", var_name="time", units=units
        )

        grid_coords = [(time_coord, 0), (new_lat_coord, 1), (new_lon_coord, 2)]

    kwargs_mod = kwargs.copy()
    if "dim_coords_and_dims" in kwargs_mod:
        del kwargs_mod["dim_coords_and_dims"]

    new_lat_coord.guess_bounds()
    new_lon_coord.guess_bounds()

    return iris.cube.Cube(
        data,
        dim_coords_and_dims=kwargs.get("dim_coords_and_dims", grid_coords),
        **kwargs_mod,
    )


def data_map_plot(
    data, lat_lims=(-90, 90), lon_lims=(-180, 180), filename=None, log=False, **kwargs
):
    """Used to plot data or an iris.cube.Cube on a map with coastlines."""
    if isinstance(data, iris.cube.Cube):
        cube = data
    else:
        cube = dummy_lat_lon_cube(data, lat_lims, lon_lims)

    cube = cube.copy()

    if "name" in kwargs:
        cube.long_name = kwargs["name"]
    else:
        cube.long_name = cube.name()

    if log:
        future_name = "log " + cube.long_name
        cube = iris.analysis.maths.log(cube)
        cube.long_name = future_name

    fig = plt.figure()
    qplt.contourf(cube)
    plt.gca().coastlines()
    if filename is not None:
        plt.savefig(filename)
    return fig


def load_cubes(files, n=None):
    """Similar to iris.load(), but seems to scale much better.

    The better scaling is partly due to the fact that this function does not try to
    merge any cubes.

    This function also solves the problem that the order in which iris.load() loads
    cubes into a CubeList is not constant, varying from one run to the next
    (presumably due to some random seed).

    """
    # Make sure files are sorted so that times increase.
    files.sort()
    cube_list = iris.cube.CubeList()
    logger.info("Loading files.")
    for f in tqdm(files[slice(0, n)]):
        cube_list.extend(iris.load(f))
    return cube_list


def lat_lon_dimcoords(latitudes, longitudes):
    """Make sure latitudes and longitudes are iris DimCoords."""
    if not isinstance(latitudes, iris.coords.DimCoord):
        latitudes = iris.coords.DimCoord(
            latitudes, standard_name="latitude", units="degrees", var_name="latitude"
        )
    if not isinstance(longitudes, iris.coords.DimCoord):
        longitudes = iris.coords.DimCoord(
            longitudes, standard_name="longitude", units="degrees", var_name="longitude"
        )
    assert_allclose(longitudes.units.modulus, 360)
    longitudes.circular = True
    return latitudes, longitudes


def lat_lon_match(
    cube,
    new_latitudes=get_centres(np.linspace(-90, 90, 721)),
    new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
):
    """Test whether regridding is necessary."""
    assert cube_contains_coords(
        cube, "latitude", "longitude"
    ), "Need [[time,] lat, lon] dimensions."

    new_latitudes, new_longitudes = lat_lon_dimcoords(new_latitudes, new_longitudes)

    for (coord_old, coord_new) in (
        (cube.coord("latitude"), new_latitudes),
        (cube.coord("longitude"), new_longitudes),
    ):
        if tuple(coord_old.points) != tuple(coord_new.points):
            break
    else:
        return True
    return False


def regrid(
    cube,
    area_weighted=None,
    new_latitudes=get_centres(np.linspace(-90, 90, 721)),
    new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
    scheme=None,
    regridder=None,
    return_regridder=False,
    verbose=False,
):
    """Regrid latitudes and longitudes.

    Expects at least latitude and longitude coordinates.

    For some input data, the `iris.analysis.Linear()` regridder yields very large
    values, which seems to be related to interpolation of masked data (at the edges of
    the domain). To combat this, the output array mask will be supplanted by one which
    masks any points outside of the original extrema.

    Args:
        cube (iris.cube.Cube): Cube to regrid.
        area_weighted (bool): If True, perform first order conservative area weighted
            regridding. If False, perform bilinear regridding. For more control, see
            `scheme` and `regridder`. If none of `area_weighted`, `scheme`, or
            `regridder` are given, nearest-neighbour regridding will be used.
        new_latitudes (array-like): New grid latitudes.
        new_longitudes (array-like): New grid longitudes.
        scheme (object with a `regridder()` method): This object should define a
            `regridder` method with interface `scheme.regridder(src_cube, target_cube)`,
            eg. `iris.analysis.Linear()`. Takes precedence over `area_weighted`.
        regridder (callable): Regridder to use. Analogous to `scheme.regridder()`,
            eg. `iris.analysis.Linear().regridder(src_cube, target_cube)`. If given,
            `regridder` needs to take an `iris.cube.Cube` as an argument and return an
            `iris.cube.Cube` on the target grid. Takes precedence over both
            `area_weighted` and `scheme`.
        return_regridder (bool): If True, return the regridder which contains the
            interpolation weights. This can be re-used for the same type of regridding
            operation between the same lat-lon grids.
        verbose (bool): Show a progress meter showing the remaining slices. Applies
            only if `cube` also has coordinates other than latitude and longitude.

    Returns:
        iris.cube.Cube: The interpolated cube. This will always contain a masked
            array even if the input data did not have a mask.
        iris Regridder: If `return_regridder` is True, the interpolated cube and
            associated regridder are returned.

    Raises:
        ValueError: If a coordinate system other than `None` or `WGS84` is
            encountered.

    Note:
        Do time coordinate differences disrupt the usage of a cached regridders?

    """
    assert cube_contains_coords(
        cube, "latitude", "longitude"
    ), "Need at least latitude and longitude coordinates."

    # TODO: Check that coordinate system discrepancies are picked up by
    # this check!!
    if lat_lon_match(cube, new_latitudes, new_longitudes):
        logger.info("No regridding needed for '{}'.".format(cube.name()))
        return cube

    # Use nearest-neighbour by default.
    if area_weighted is None and scheme is None and regridder is None:
        scheme = iris.analysis.Nearest()

    logger.debug("Regridding '{}'.".format(cube.name()))

    new_latitudes, new_longitudes = lat_lon_dimcoords(new_latitudes, new_longitudes)

    if len(cube.shape) > 2:
        # Call the regridding function recursively with slices of the
        # data, in order to try to prevent occasional Segmentation Faults
        # that occur when trying to regrid a large chunk of data in > 2
        # dimensions.

        # Make sure that the latitude and longitude coordinates are placed after the
        # initial coordinates to ensure proper indexing below. Note that additional
        # coordinates may exist which are not reflected in the data's shape - thus
        # the use of `len(cube.shape) - 1` as opposed to simply `-1`.
        assert (
            set(
                (
                    coord.name()
                    for coord in (
                        cube.coords()[len(cube.shape) - 2],
                        cube.coords()[len(cube.shape) - 1],
                    )
                )
            )
            == set(("latitude", "longitude"))
        )

        # Ensure the initial coordinates reflect the data.
        assert all(
            (len(cube.coords()[i].points) == cube.shape[i])
            for i in range(len(cube.shape))
        )

        # Iterate over all dimensions but (guaranteed to be preceding) latitude and
        # longitude.
        regridded_cubes = iris.cube.CubeList()
        indices_lists = [
            ind_arr.flatten()
            for ind_arr in np.indices(cube.shape[: len(cube.shape) - 2])
        ]
        for indices in tqdm(
            zip(*indices_lists),
            total=len(indices_lists[0]),
            desc="Regridding time slices",
            disable=not verbose,
        ):
            # Reuse the regridder between subsequent regridding operations.
            regridded_cube, regridder = regrid(
                cube[indices],
                area_weighted=area_weighted,
                new_latitudes=new_latitudes,
                new_longitudes=new_longitudes,
                scheme=scheme,
                regridder=regridder,
                return_regridder=True,
            )
            regridded_cubes.append(regridded_cube)

        to_remove = set(
            [coord.name() for coord in regridded_cubes[0].coords()]
        ).intersection(set(("year", "month_number")))
        for regridded_cube in regridded_cubes:
            for coord in to_remove:
                regridded_cube.remove_coord(coord)
        if return_regridder:
            return regridded_cubes.merge_cube(), regridder
        return regridded_cubes.merge_cube()

    # This is where the core 2D regridding takes place.
    assert cube_contains_coords(
        cube, "latitude", "longitude"
    ), "Need [lat, lon] dimensions for core algorithm."

    WGS84 = iris.coord_systems.GeogCS(
        semi_major_axis=6378137.0, semi_minor_axis=6356752.314245179
    )
    # Make sure coordinate systems are uniform.
    systems = [
        cube.coord(coord_name).coord_system for coord_name in ["latitude", "longitude"]
    ]
    assert systems[0] == systems[1]

    if systems[0] is None:
        coord_sys = None
    elif (systems[0].semi_major_axis == WGS84.semi_major_axis) and (
        systems[0].semi_minor_axis == WGS84.semi_minor_axis
    ):
        logger.debug("Using WGS84 coordinate system for regridding.")
        coord_sys = WGS84
        # Fix floating point 'bug' where the inverse flattening of the
        # coord system that comes with the dataset does not match the
        # inverse flattening that is calculated by iris upon giving the two
        # axis parameters above (which do match between the two coordinate
        # systems). Inverse flattening calculated by iris:
        # 298.2572235629972, vs that in the Copernicus_SWI dataset:
        # 298.257223563, which seems like it is simply truncated.
        for coord_name in ["latitude", "longitude"]:
            cube.coord(coord_name).coord_system = WGS84
    else:
        raise ValueError("Unknown coord_system:{:}".format(systems[0]))

    for coord in [c for c in cube.coords() if c.name() in ["latitude", "longitude"]]:
        if not coord.has_bounds():
            coord.guess_bounds()

    if regridder is None:
        logger.debug("Constructing regridder.")
        for coord in [new_latitudes, new_longitudes]:
            coord.coord_system = coord_sys

        grid_coords = [(new_latitudes, 0), (new_longitudes, 1)]

        new_grid = iris.cube.Cube(
            np.zeros([coord[0].points.size for coord in grid_coords]),
            dim_coords_and_dims=grid_coords,
        )

        for coord in new_grid.coords():
            if not coord.has_bounds():
                coord.guess_bounds()

        if scheme is None:
            if area_weighted:
                regridder = iris.analysis.AreaWeighted(mdtol=1).regridder(
                    cube, new_grid
                )
            else:
                regridder = iris.analysis.Linear().regridder(cube, new_grid)
        else:
            regridder = scheme.regridder(cube, new_grid)
    else:
        logger.debug("Using given regridder.")

    logger.debug("Cube has lazy data: {}.".format(cube.has_lazy_data()))
    interpolated_cube = homogenise_cube_mask(regridder(cube))

    if isinstance(regridder, iris.analysis._regrid.RectilinearRegridder):
        # Fix the extreme values that can occur with the
        # `iris.analysis.Linear()` regridding scheme.

        omax = float(cube.collapsed(cube.dim_coords, iris.analysis.MAX).data)
        omin = float(cube.collapsed(cube.dim_coords, iris.analysis.MIN).data)
        range_thres = omax - omin

        # Heuristic approach to filtering extreme values.
        extr_mask = (interpolated_cube.data > (omax + range_thres)) | (
            interpolated_cube.data < (omin - range_thres)
        )
        if np.any(extr_mask):
            logger.warning(
                f"Masking {np.sum(extr_mask)} regridded values that exceeded the "
                f"threshold range for cube {cube.name()}."
            )
            interpolated_cube.data.mask |= extr_mask

    if return_regridder:
        return interpolated_cube, regridder
    return interpolated_cube


@dataset_cache
def regrid_dataset(
    dataset,
    new_latitudes=get_centres(np.linspace(-90, 90, 721)),
    new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
    **kwargs,
):
    logger.debug(f"Regridding '{dataset}' with {len(dataset)} variable(s).")
    regridded_cubes = iris.cube.CubeList(
        [
            regrid(
                cube,
                new_latitudes=new_latitudes,
                new_longitudes=new_longitudes,
                **kwargs,
            )
            for cube in dataset.cubes
        ]
    )
    return regridded_cubes


def _temporal_nn(source_data, target_index, interpolate_mask, n_months, verbose=False):
    """Temporal NN interpolation of a single month.

    Args:
        source_data ((s, m, n) array of float): Source data to use for the interpolation.
        target_index (int): Temporal index (axis=0) corresponding to the timestep to
            fill using interpolation.
        interpolate_mask ((m, n) array of bool): Boolean mask that is true where
            interpolation should take place.
        n_months (int): Number of months to look forwards and backwards to find valid
            data. The total number of months selected for each interpolation step will
            be `2 * n_months + 1` (including the current timestep).
        verbose (bool): If True, show progress bar.

    Returns:
        numpy masked array: (N,) masked array containing the interpolated values for
            each of the `N` True elements in `interpolate_mask`.

    """
    n_interp = interpolate_mask.sum()
    monthly_target_data = np.ma.MaskedArray(np.empty(n_interp), mask=True)

    for i, indices in enumerate(
        tqdm(
            zip(*np.where(interpolate_mask)),
            total=n_interp,
            leave=False,
            disable=not verbose,
        )
    ):
        adjacent_data = source_data[
            (slice(target_index, target_index + 2 * n_months + 1), *indices)
        ]
        assert adjacent_data.mask[n_months], "Data to interpolate must be masked."

        # Try to find at least one match in the fewest months possible.
        for d in range(1, n_months + 1):
            selection_mask = (
                adjacent_data.mask[n_months - d],
                adjacent_data.mask[n_months + d],
            )
            if all(selection_mask):
                # All data is masked, so there is no valid data to choose from.
                continue
            selection = np.ma.MaskedArray(
                [adjacent_data[n_months - d], adjacent_data[n_months + d]]
            )
            # Fill in the missing element.
            monthly_target_data[i] = np.mean(selection)
            # Stop looking for matches.
            break
    return monthly_target_data


@dataset_cache
def temporal_nn(
    dataset, target_timespan, n_months, mask=None, threshold=0, verbose=False
):
    """Temporal NN interpolation of missing data.

    Args:
        dataset (wildfires.data.Dataset): Dataset containing a single cube to
            interpolate.
        target_timespan (tuple of datetime): Start and end datetimes between which to
            interpolate.
        n_months (int): Number of months to look forwards and backwards to find valid
            data. The total number of months selected for each interpolation step will
            be `2 * n_months + 1` (including the current timestep).
        mask ((m, n) array of bool): Mask that is True where interpolation should occur.
            If None, interpolation is not carried out over water and south of 60
            degrees latitude.
        threshold (float): Threshold in [0, 1] denoting the minimum fraction (not
            inclusive) of missing data in `source_masks` for which to carry out
            interpolation. Larger thresholds can be used to restrict interpolation
            geographically to regions with poor data availability.
        verbose (bool or int): If True, show progress bars. Giving 0 is equivalent to
            giving False (i.e. no progress bars), giving 1 shows a progress bar for
            the individual months, while giving 2 shows an additional progress bar for
            the individual samples within each month.

    Returns:
        iris CubeList: CubeList containing the interpolated data for the time period
            `target_timespan` in a single iris Cube.

    Raises:
        ValueError: If `dataset` does not contain exactly 1 cube.
        ValueError: If `n_months` is not an integer.
        RuntimeError: If there was insufficient data to satisfy either the target or
            the source time period.

    """
    if len(dataset.cubes) != 1:
        raise ValueError(f"Expected 1 cube, got {len(dataset.cubes)}.")

    if not isinstance(n_months, (int, np.integer)):
        raise ValueError(
            f"`n_months` should be an integer. Got type '{type(n_months)}'."
        )

    if isinstance(verbose, int):
        m_verbose = False  # Month verbosity.
        s_verbose = False  # Sample verbosity.
        if verbose > 0:
            m_verbose = True
        if verbose > 1:
            s_verbose = True
    elif verbose:
        m_verbose = True
        s_verbose = True

    # Set up data.
    source = dataset.copy(deep=False)
    target = dataset.copy(deep=True)

    # Discard unneeded months.
    source_timespan = (
        target_timespan[0] - relativedelta(months=n_months),
        target_timespan[1] + relativedelta(months=n_months),
    )
    source.limit_months(*source_timespan)
    target.limit_months(*target_timespan)

    target_months = (
        (target_timespan[1].year - target_timespan[0].year) * 12
        + target_timespan[1].month
        - target_timespan[0].month
    )

    source_months = target_months + n_months * 2

    # Sanity check.
    assert source_months == (
        (source_timespan[1].year - source_timespan[0].year) * 12
        + source_timespan[1].month
        - source_timespan[0].month
    )

    # Check that a sufficient number of months were present.
    if not target.cube.shape[0] == (target_months + 1):
        raise RuntimeError("Missing data for the target time period.")
    if not source.cube.shape[0] == (source_months + 1):
        raise RuntimeError("Missing data for the source time period.")

    if not target.cube.coords("month_number"):
        iris.coord_categorisation.add_month_number(target.cube, "time")
    target.homogenise_masks()

    # Set up the source mask (where to interpolate).
    if mask is None:
        # Add the land mask.
        mask = match_shape(
            get_land_mask(n_lon=dataset.cube.shape[-1]), dataset.cube.shape[1:]
        )
        # Ignore regions south of -60° S.
        mask &= match_shape(
            box_mask(lats=(-60, 90), lons=(-180, 180), n_lon=dataset.cube.shape[-1]),
            mask.shape,
        )

    interpolate_masks = {}
    for month_number in range(1, 13):
        single_months = target.cube.extract(iris.Constraint(month_number=month_number))
        missing_fraction = np.mean(single_months.data.mask & mask, axis=0)
        interpolate_masks[month_number] = missing_fraction > threshold

    # Iterate over the months to fill.
    current = target_timespan[0]
    for target_index in tqdm(range(target_months), disable=not m_verbose):
        month_number = current.month
        interpolate_mask = (
            interpolate_masks[month_number]
            & source.cube.data.mask[target_index + n_months]
        )
        target.cube.data[target_index][interpolate_mask] = _temporal_nn(
            source.cube.data,
            target_index,
            interpolate_mask,
            n_months,
            verbose=s_verbose,
        )
        current += relativedelta(months=1)

    return target.cubes


def _persistent_gap_filling(cube, thres=0.5, verbose=False):
    """Fill gaps >= (thres * 100)% of months with minimum value at that location.

    This is done in-place.

    """
    if not cube.coords("month_number"):
        iris.coord_categorisation.add_month_number(cube, "time")

    combined_mask = np.all(cube.data.mask, axis=0)

    nr_inval_cube = cube.copy(
        data=np.ma.MaskedArray(
            cube.data.mask, mask=match_shape(combined_mask, cube.shape)
        )
    )

    min_cube = cube.collapsed("time", iris.analysis.MIN)

    # Month numbers in [1, 12].
    month_numbers = cube.coord("month_number").points

    for month_number in tqdm(range(1, 13), desc="Months", disable=not verbose):
        extracted = iris.Constraint(month_number=month_number).extract(nr_inval_cube)
        missing_frac = np.sum(extracted.data, axis=0) / extracted.shape[0]
        persistent = ((missing_frac + 1e-5) >= thres).data
        persistent[combined_mask] = False

        for month_index in np.where(month_numbers == month_number)[0]:
            month_data = cube.data[month_index]

            fill_mask = persistent & cube.data.mask[month_index]
            month_data[fill_mask] = min_cube.data[fill_mask]

            cube.data[month_index] = month_data

    return cube


def _harmonic_fit(t, params):
    """Sine-based fitting including offset.

    Args:
        t (int): Time index.
        params (array-like):
            0th - offset
            1th - gradient
            (2j, 2j+1) entries - jth component amplitude and phase, j <= 1.

    Returns:
        float: Fitted function value at `t`.

    """
    t = np.asarray(t, dtype=np.float64)
    output = np.zeros_like(t, dtype=np.float64)
    output += params[0]
    output += params[1] * t
    for (j, (amplitude, phase)) in enumerate(zip(params[2::2], params[3::2])):
        j += 1
        output += amplitude * np.sin((2 * np.pi * j * t / 12) + phase)
    return output


def _min_fit(x, *args):
    """Function to be minimised.

    Args:
        x (array-like): Fit parameters.
        args: Month indices and corresponding data to fit to.

    Returns:
        float: MSE fit error.

    """
    ts = args[0]
    fit_data = args[1]
    return np.sum((fit_data - _harmonic_fit(ts, x)) ** 2.0)


def _season_fill(fill_locs, data, k):
    ts = np.arange(data.shape[0])

    for xi, yi in zip(*np.where(fill_locs)):
        sel = data[:, xi, yi]
        # Execute minimisation.
        res = minimize(_min_fit, np.zeros(2 * k + 2), (ts, sel))
        # Replace masked elements with function fit values.
        sel[sel.mask] = _harmonic_fit(ts, res.x)[sel.mask]

    return data


def _season_model_filling(cube, k=4, ncpus=1, verbose=False):
    """Season-trend filling in-place."""
    # Fill where there is some valid data, but not only valid data, since there would
    # be nothing to fill in the latter case.
    fill_locs = np.any(~cube.data.mask, axis=0) & (~np.all(~cube.data.mask, axis=0))

    # Partition the rows of the array into chunks to be processed.
    chunk_edges = np.unique(
        np.append(np.arange(0, cube.shape[1], 2, dtype=np.int64), cube.shape[1])
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
        fs = []
        processed_slices = []
        for chunk_s, chunk_e in zip(chunk_edges[:-1], chunk_edges[1:]):
            chunk_slice = slice(chunk_s, chunk_e)
            if not np.any(fill_locs[chunk_slice]):
                # Skip those slices without anything to fill.
                continue
            processed_slices.append(chunk_slice)
            fs.append(
                executor.submit(
                    _season_fill, fill_locs[chunk_slice], cube.data[:, chunk_slice], k
                )
            )

        for f in tqdm(
            concurrent.futures.as_completed(fs),
            desc="Season model filling",
            total=len(fs),
            disable=not verbose,
        ):
            pass

        for f, chunk_slice in zip(fs, processed_slices):
            cube.data[:, chunk_slice] = f.result()

    return cube


@dataset_cache
def persistent_season_trend_fill(
    dataset, target_timespan=None, persistent_perc=50, k=4, verbose=False, ncpus=None
):
    """Interpolation of missing data using minimum values and season-trend model.

    First, persistent gaps are filled using the minimum value observed at that
    location. This is followed by the fitting of a season-trend model to fill the
    remaining missing values.

    Args:
        target_timespan (tuple of datetime or None): Start and end datetimes
            between which to interpolate. If None, the current temporal range of
            data will be used.
        persistent_perc (int in [0, 100]): Percentage of data that needs to be
            missing for a given month at a given location for the month to be
            considered affected by a persistent gap.
        k (int): Number of harmonic terms used in the season-trend model.
        verbose (bool): If True, show progress bars.
        ncpus (int or None): Number of processes to use for the season-trend model
            fitting. If None, `wildfires.qstat.get_ncpus()` will be used.

    Returns:
        iris CubeList: CubeList containing the interpolated data for the time period
            `target_timespan` in a single iris Cube.

    Raises:
        ValueError: If `dataset` does not contain exactly 1 cube.
        ValueError: If `persistent_perc` is not an integer.
        ValueError: If `k` is not an integer.

    """
    if len(dataset.cubes) != 1:
        raise ValueError(f"Expected 1 cube, got {len(dataset.cubes)}.")

    if not isinstance(persistent_perc, (int, np.integer)):
        raise ValueError(
            f"`persistent_perc` should be an integer. "
            f"Got type '{type(persistent_perc)}'."
        )

    if not isinstance(k, (int, np.integer)):
        raise ValueError(f"`k` should be an integer. Got type '{type(k)}'.")

    if ncpus is None:
        ncpus = get_ncpus()

    # Set up data.
    target = dataset.copy(deep=True)
    if target_timespan is not None:
        target.limit_months(*target_timespan)

    if not target.cube.coords("month_number"):
        iris.coord_categorisation.add_month_number(target.cube, "time")
    target.homogenise_masks()

    _season_model_filling(
        _persistent_gap_filling(
            target.cube, thres=persistent_perc / 100, verbose=verbose
        ),
        k=k,
        ncpus=ncpus,
        verbose=verbose,
    )

    return target.cubes


regrid_dataset.__doc__ = "Dataset wrapper\n" + regrid.__doc__


def monthly_constraint(
    t,
    time_range=(PartialDateTime(2000, 1), PartialDateTime(2010, 1)),
    inclusive_lower=True,
    inclusive_upper=True,
):
    """Constraint function which ignores the day and only considers the
    year and month.

    """
    lower_op = operator.ge if inclusive_lower else operator.gt
    upper_op = operator.le if inclusive_upper else operator.lt
    comp_datetime = PartialDateTime(year=t.year, month=t.month)

    return lower_op(comp_datetime, time_range[0]) and upper_op(
        comp_datetime, time_range[1]
    )


class RegisterDatasets(ABCMeta):
    """Datasets are registered into a central list to keep track of them.

    If any subclass (dataset) is used as another class's superclass, it is removed
    from the registry in favour of its subclasses.

    """

    datasets = None

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if cls.datasets is None:
            cls.datasets = set()
        cls.datasets.add(cls)
        cls.datasets -= set(bases)

    def __iter__(cls):
        return iter(cls.datasets)

    def __str__(cls):
        if cls in cls.datasets:
            return cls.__name__
        return f"{cls.__name__}: {', '.join(map(str, cls))}"


class Dataset(metaclass=RegisterDatasets):
    # TODO: Make sure these get overridden by the subclasses, or that every
    # dataset uses these consistently (if defining custom date coordinates).
    calendar = "gregorian"
    time_unit_str = "days since 1970-01-01 00:00:00"
    time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

    _pretty = None
    # Override the `pretty_variable_names` dictionary in each class where bespoke
    # pretty variable names are desired. Keys are the raw variables names.
    pretty_variable_names = dict()

    @abstractmethod
    def __init__(self):
        """Instantiate the dataset by loading Iris Cubes."""

    def __str__(self):
        return "{} ({}, {}, {})".format(
            self.name, self.min_time, self.max_time, self.frequency
        )

    def __repr__(self):
        return str(self) + " at {}".format(id(self))

    def __eq__(self, other):
        """Equality testing that ignores data values and only looks at metadata.

        The equality is derived solely from the `self.cubes` attribute, and the
        coordinates and metadata of each cube in this CubeList. This means that
        changes in the stored data are ignored!

        """
        if isinstance(other, Dataset):
            return self._shallow == other._shallow
        return NotImplemented

    def __len__(self):
        return len(self.cubes)

    def __getitem__(self, index):
        """Return a subset or single cube of `self`.

        Args:
            index (slice, str, int): Index used to retrieve (on or more) items see
                `Returns` section below.

        Returns:
            `Dataset`: If `index` is a slice object, a subset of `self` containing the
                cubes at the indices included in the slice will be returned.
            `iris.cube.Cube`: If `index` is an integer or a string, the corresponding
                cube will be returned, if found. Matching works with pretty and raw
                names.

        Raises:
            VariableNotFoundError: If `index` is an integer or string, but the
                specified cube could not be found.

        """
        if isinstance(index, slice):
            new_dataset = self.copy(deep=False)
            new_dataset.cubes = self.cubes[index]
            return new_dataset
        try:
            if isinstance(index, str):
                # Substitute pretty name for raw name if needed.
                index = self._get_raw_variable_names().get(index, index)
                new_index = self.variable_names(which="raw").index(index)
            else:
                new_index = index
            return self.cubes[new_index]
        except ValueError as exc:
            error_msg = strip_multiline(
                f"""No cube could be found for index '{index}'.
                Available: integer indices {list(range(len(self.cubes)))},
                raw names {self.variable_names(which="raw")},
                or pretty names {self.variable_names(which="pretty")}."""
            )
            raise VariableNotFoundError(error_msg) from exc

    @classmethod
    def _get_raw_variable_names(cls):
        """The inverse of the pretty variable name dict."""
        all_pretty = list(cls.pretty_variable_names.values())
        assert len(set(all_pretty)) == len(
            all_pretty
        ), "Mapping pretty to raw names requires unique pretty variable names."
        return dict((pretty, raw) for raw, pretty in cls.pretty_variable_names.items())

    @property
    def _shallow(self):
        """Create a hashable shallow description of the CubeList.

        Note:
            Only metadata and coordinates are considered.

        Returns:
            str

        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=((r".*guessing contiguous bounds."))
            )
            # Join up the elements in this list later to construct the string.
            cubelist_hash_items = []
            for cube in self.cubes:
                # Compute each coordinate's string representation (the coord's __hash__
                # attribute only returns the output of id()).
                for coord in cube.coords():
                    point_diffs = np.diff(coord.points)
                    unique_diffs = np.unique(point_diffs)
                    # TODO: More robust comparison of numpy arrays compared to
                    # just converting them to strings, eg. using rounding!
                    if len(unique_diffs) != 1:
                        point_diff_str = str(tuple(point_diffs))
                    else:
                        point_diff_str = str(unique_diffs[0])

                    cubelist_hash_items += list(
                        map(
                            str,
                            (
                                tuple(sorted(coord.attributes.items())),
                                coord.circular if hasattr(coord, "circular") else None,
                                tuple(coord.contiguous_bounds())
                                if coord.is_monotonic()
                                and coord.contiguous_bounds() is not None
                                else None,
                                coord.coord_system,
                                coord.dtype,
                                coord.has_bounds(),
                                coord.is_contiguous(),
                                coord.is_monotonic(),
                                coord.long_name,
                                coord.name(),
                                np.min(coord.points),
                                np.max(coord.points),
                                np.mean(coord.points),
                                point_diff_str,
                                coord.shape,
                                coord.standard_name,
                                coord.var_name,
                            ),
                        )
                    )
                for key, value in sorted(cube.metadata._asdict().items()):
                    # This contains the data for the `cube.attributes` property.
                    if key == "attributes":
                        # Get string representation of the attributes dictionary.
                        cubelist_hash_items += [
                            str(
                                tuple(
                                    (attribute, attribute_value)
                                    for attribute, attribute_value in sorted(
                                        value.items()
                                    )
                                )
                            )
                        ]

                    else:
                        assert not isinstance(value, dict), (
                            "Dicts should be handled specially as above to maintain "
                            "consistent sorting."
                        )
                        cubelist_hash_items += [str(key) + str(value)]
        return "\n".join(cubelist_hash_items)

    def __check_cubes(self):
        """Verification functions that should be run prior to accessing data.

        Cubes are sorted by name and the uniqueness of the variables names is
        verified.

        """
        self.__cubes = iris.cube.CubeList(
            sorted(self.__cubes, key=lambda cube: cube.name())
        )

        raw_names = tuple(cube.name() for cube in self.__cubes)
        all_names = []
        for raw_name in raw_names:
            all_names.append(raw_name)
            if raw_name in self.pretty_variable_names:
                all_names.append(self.pretty_variable_names[raw_name])

        assert len(set(all_names)) == len(
            all_names
        ), "All variable names should be unique."

        for cube in self.__cubes:
            n_dim = len(cube.shape)
            coord_names = []
            if (
                hasattr(self, "_special_coord_cubes")
                and cube.name() in self._special_coord_cubes
            ):
                coord_names.extend(self._special_coord_cubes[cube.name()])
            elif n_dim == 2:
                coord_names.extend(("latitude", "longitude"))
            elif n_dim == 3:
                coord_names.extend(("time", "latitude", "longitude"))
            elif n_dim > 3:
                warnings.warn(
                    f"\n{cube}\nin '{type(self)}' at '{id(self)}' has {n_dim} axes "
                    "with unexpected coordinate names."
                )

            for coord_name in coord_names:
                try:
                    coord = cube.coord(coord_name)
                    coord.var_name = coord_name
                    if coord_name in ("latitude", "longitude"):
                        # Check that the coordinates are monotonically increasing.
                        assert np.all(
                            np.diff(coord.points) > 0
                        ), f"{coord_name.capitalize()}s need to increase monotonically."
                        if coord_name == "longitude":
                            # Check that longitudes are in the [-180, 180] system.
                            assert not (
                                in_360_longitude_system(coord.points)
                            ), "Longitudes need to be in the [-180, 180] system."

                except iris.exceptions.CoordinateNotFoundError:
                    warnings.warn(
                        f"{n_dim}-dimensional cube '{cube}' in '{self.name}' did not "
                        f"have a '{coord_name}' coordinate."
                    )

            # NOTE: This step is necessary to remove discrepancies between cubes
            # before and after saving & loading them using iris.save() & iris.load(),
            # which seems to change key attributes, like 'Conventions', from CF-1.4 to
            # CF-1.5, for example.
            cube.attributes["Conventions"] = "CF-1.5"

    @property
    def cubes(self):
        # This might happen when assigning self.cubes to the result of
        # self.read_cache(), for example.
        # FIXME: Resolve this hack by changing the way the result of self.read_cache()
        # FIXME: is used.
        if self.__cubes is None:
            logger.warning("Cubes is None.")
            return None
        self.__check_cubes()
        return self.__cubes

    @cubes.setter
    def cubes(self, new_cubes):
        """Assign new cubes.

        Raises:
            NonUniformCoordError: If one or more coordinates are not uniform.

        """
        # This might happen when assigning self.cubes to the result of
        # self.read_cache(), for example.
        # FIXME: Resolve this hack by changing the way the result of self.read_cache()
        # FIXME: is used.
        if new_cubes is None:
            logger.warning(f"Assigning None to cubes.")
            self.__cubes = None
        else:
            assert isinstance(
                new_cubes, iris.cube.CubeList
            ), "New cube list must be an iris CubeList (`iris.cube.CubeList`)."

            # Ensure uniformity of latitudes and longitudes. They should both be
            # monotonically increasing, and the longitudes should be in the
            # [-180, 180] system.
            for i in range(len(new_cubes)):
                # Ensure the proper longitude system.
                if new_cubes[i].coords("longitude"):
                    if in_360_longitude_system(new_cubes[i].coord("longitude").points):
                        # Reorder longitudes into the [-180, 180] system.
                        tr_longitudes, tr_indices = translate_longitude_system(
                            new_cubes[i].coord("longitude").points, return_indices=True
                        )
                        new_cubes[i] = reorder_cube_coord(
                            new_cubes[i], tr_indices, tr_longitudes, name="longitude"
                        )
                    new_cubes[i].coord("longitude").circular = True

                # Ensure longitudes and latitudes are properly ordered.
                for coord in ("latitude", "longitude"):
                    if new_cubes[i].coords(coord):
                        if not np.all(np.diff(new_cubes[i].coord(coord).points) > 0):
                            # If the coordinate is not monotonically increasing we
                            # need to handle this.
                            if np.all(np.diff(new_cubes[i].coord(coord).points) < 0):
                                # If they are monotonically decreasing we just need the flip them.
                                logger.debug(
                                    f"Inverting {coord}s for: {new_cubes[i].name()}."
                                )
                                lat_index = [
                                    coord.name() for coord in new_cubes[i].coords()
                                ].index(coord)
                                slices = [
                                    slice(None) for i in range(len(new_cubes[i].shape))
                                ]
                                slices[lat_index] = slice(None, None, -1)
                                new_cubes[i] = new_cubes[i][tuple(slices)]
                            else:
                                # If there is another pattern, one could attempt
                                # regridding, but we will alert the user to this
                                # instead.
                                raise NonUniformCoordError(
                                    f"{coord.capitalize()}s for {new_cubes[i].name()} are not "
                                    "uniform."
                                )

            self.__cubes = new_cubes

    @property
    def cube(self):
        """Convenience method to access a single stored cube."""
        if len(self.cubes) != 1:
            raise ValueError(f"Expected 1 cube, but found {len(self.cubes)} cubes.")
        return self.cubes[0]

    def copy_cubes_no_data(self):
        """Copy everything except the cube data.

        This includes cube metadata.

        Returns:
            `Dataset`: The copied dataset.

        """
        new_cubelist = iris.cube.CubeList()
        for cube in self:
            # Copy everything about the cubes but their data.
            new_cubelist.append(cube.copy(data=cube.core_data()))

        dataset = copy(self)
        dataset.cubes = new_cubelist
        return dataset

    def copy(self, deep=False):
        """Make a copy.

        Args:
            deep (bool): If False (default), create a shallow copy which will copy the
                cube list but not the underlying cubes. If True, create a deep copy of
                everything including the underlying cubes and their data.

        Returns:
            `Dataset`: The copy.

        """
        if deep:
            return deepcopy(self)
        dataset = copy(self)
        dataset.cubes = copy(self.cubes)
        return dataset

    def homogenise_masks(self):
        for i, cube in enumerate(self):
            self.cubes[i] = homogenise_cube_mask(cube)

    def apply_masks(self, *masks):
        """Apply given masks on top of existing masks inplace."""
        # Ensure masks are recorded in a format to enable the modifications below.
        self.homogenise_masks()
        # Check the given masks.
        # TODO: If the masks contain cubes, check that their coordinates are
        # TODO: consistent.
        masks = list(masks)
        for i, mask in enumerate(masks):
            if isinstance(mask, iris.cube.Cube):
                masks[i] = mask.data

        for cube in self:
            # Create the combined mask and apply them each in turn.
            # TODO: Only calculate the aggregated mask for each unique shape present.
            cube.data.mask |= reduce(
                np.logical_or, (match_shape(mask, cube.shape) for mask in masks)
            )
        return self

    def grid(self, coord="latitude"):
        try:
            diffs = np.diff(self[0].coord(coord).points)
            mean = np.mean(diffs)
            if np.all(np.isclose(diffs, mean)):
                return np.abs(mean)

        except iris.exceptions.CoordinateNotFoundError:
            pass

        return "N/A"

    @property
    def lat_grid(self):
        return self.grid("latitude")

    @property
    def lon_grid(self):
        return self.grid("longitude")

    @property
    def _temporal_cubes(self):
        temporal_cubes = iris.cube.CubeList()
        for cube in self:
            if any(coord.name() == "time" for coord in cube.coords()):
                temporal_cubes.append(cube)
        return temporal_cubes

    @property
    def frequency(self):
        temporal_cubes = self._temporal_cubes
        if temporal_cubes:
            time_coord = temporal_cubes[0].coord("time")
            if len(time_coord.points) == 1:
                return "static"
            raw_start = time_coord.cell(0).point
            raw_end = time_coord.cell(1).point
            start = datetime(raw_start.year, raw_start.month, 1)
            end = datetime(raw_end.year, raw_end.month, 1)
            if (start + relativedelta(months=+1)) == end:
                return "monthly"
            if (start + relativedelta(months=+12)) == end:
                return "yearly"
            month_number_coords = temporal_cubes[0].coords("month_number")
            if month_number_coords:
                assert len(month_number_coords) == 1
                if tuple(month_number_coords[0].points) == tuple(range(1, 13)):
                    return "climatology"
            return str(raw_end - raw_start)
        else:
            return "static"

    @property
    def min_time(self):
        temporal_cubes = self._temporal_cubes
        if temporal_cubes:
            return ensure_datetime(
                max(cube.coord("time").cell(0).point for cube in temporal_cubes)
            )
        else:
            return "static"

    @property
    def max_time(self):
        temporal_cubes = self._temporal_cubes
        if temporal_cubes:
            return ensure_datetime(
                min(cube.coord("time").cell(-1).point for cube in temporal_cubes)
            )
        else:
            return "static"

    @property
    def name(self):
        return type(self).__name__

    def names(self, which="all", squeeze=True):
        if which == "all":
            return (self.name, self.pretty)
        if which == "raw":
            if squeeze:
                return self.name
            return (self.name,)
        if which == "pretty":
            if squeeze:
                return self.pretty
            return (self.pretty,)
        raise ValueError("Unknown format: '{}'.".format(which))

    @property
    def pretty(self):
        if self._pretty is None:
            return self.name
        return self._pretty

    @pretty.setter
    def pretty(self, value):
        self._pretty = value

    def variable_names(self, which="all"):
        raw_names = tuple(cube.name() for cube in self.cubes)

        if which == "all":
            all_names = []
            for raw_name in raw_names:
                all_names.append(
                    (raw_name, self.pretty_variable_names.get(raw_name, raw_name))
                )
            return tuple(all_names)
        if which == "raw":
            return raw_names
        if which == "pretty":
            pretty_names = []
            for raw_name in raw_names:
                pretty_names.append(self.pretty_variable_names.get(raw_name, raw_name))
            return tuple(pretty_names)
        raise ValueError("Unknown format: '{}'.".format(which))

    @property
    def cache_filename(self):
        return os.path.join(DATA_DIR, "cache", self.name + ".nc")

    @classmethod
    def _get_cache_filename(cls):
        return os.path.join(DATA_DIR, "cache", cls.__name__ + ".nc")

    @staticmethod
    def save_data(cache_data, target_filename):
        """Save as NetCDF file.

        Args:
            cache_data (iris.cube.Cube or iris.cube.CubeList): This will be
                saved as a NetCDF file.
            target_filename (str): The filename that the data will be saved
                to. Must end in '.nc', since the data is meant to be saved
                as a NetCDF file.

        Returns:
            str or None: The current hex commit sha hash of the repo if a new file was
            created (or the current package version). Otherwise, if the file was
            already there and not overwritten, None is returned.

        """
        assert (
            target_filename[-3:] == ".nc"
        ), "Data must be saved as a NetCDF file, got:'{:}'".format(target_filename)
        assert isinstance(
            cache_data, (iris.cube.Cube, iris.cube.CubeList)
        ), f"Data to be saved must either be a Cube or a CubeList. Got: {cache_data}."

        if isinstance(cache_data, iris.cube.Cube):
            cache_data = iris.cube.CubeList([cache_data])

        if os.path.isfile(target_filename):
            # TODO: Want to overwrite if the commit hash is different?
            # Maybe add a flag to do this.
            logger.info("File exists, not overwriting:'{:}'".format(target_filename))
        else:
            if repo is not None:
                assert (not repo.untracked_files) and (
                    not repo.is_dirty()
                ), "All changes must be committed and all files must be tracked."
                repo_commit = repo.head.ref.commit.hexsha
            else:
                # Simply use the package version otherwise.
                repo_commit = __version__

            # Note down the commit sha hash so that the code used to
            # generate the cached data can be retrieved easily later on.
            for cube in cache_data:
                cube.attributes["commit"] = repo_commit

            if not os.path.isdir(os.path.dirname(target_filename)):
                os.makedirs(os.path.dirname(target_filename))
            logger.info("Realising data.")
            cache_data.realise_data()
            logger.info("Saving cubes to:'{:}'".format(target_filename))
            iris.save(cache_data, target_filename, zlib=False)
            return cube.attributes["commit"]

    @staticmethod
    def read_data(target_filename):
        """Read from NetCDF file.

        Args:
            target_filename (str): The filename that the data will be saved to. Must
                end in '.nc', since the data is meant to be saved as a NetCDF file.

        Raises:
            CommitMatchError: If the commit hashes of the cubes that are loaded do not
                match.

        """
        if os.path.isfile(target_filename):
            cubes = iris.load(target_filename)
            if not cubes:
                os.remove(target_filename)
                logger.warning(
                    "No cubes were found. Deleted file:{:}".format(target_filename)
                )
                return

            commit_hashes = [cube.attributes["commit"] for cube in cubes]

            if not len(set(commit_hashes)) == 1:
                raise CommitMatchError("Cubes do not stem from the same commit.")

            logger.debug("Returning cubes from:'{:}'".format(target_filename))
            return cubes
        else:
            logger.info("File does not exist:'{:}'".format(target_filename))

    def write_cache(self):
        """Write list of cubes to disk as a NetCDF file using iris.

        Also record the git commit id that the data was generated with,
        making sure that there are no uncommitted changes in the repository
        at the time.

        """
        self.__check_cubes()
        self.save_data(self.cubes, self.cache_filename)

    def read_cache(self):
        cubes = self.read_data(self.cache_filename)
        if cubes:
            self.cubes = cubes
            logger.info(
                "Returning cubes from:'{:}' -> Dataset "
                "timespan {:} -- {:}. Generated using commit {:}".format(
                    self.cache_filename,
                    self.min_time,
                    self.max_time,
                    self.cubes[0].attributes["commit"],
                )
            )
            return self.cubes

    def select_data(
        self, latitude_range=(-90, 90), longitude_range=(-180, 180), inplace=False
    ):
        """Select a geographical sub-set of the original data for each cube.

        Args:
            latitude_range (tuple of float): Latitude range to select.
            longitude_range (tuple of float): Longitude range to select.
            inplace (bool): If True, subset the cubes inplace without returning a copy
                of the selected data.

        Returns:
            `Dataset`: Dataset with cubes containing only data for the selected
                region. Depending on `inplace`, this may be a copy of the original
                dataset.

        """

        def geo_subset(cube):
            return cube.intersection(latitude=latitude_range).intersection(
                longitude=longitude_range
            )

        if inplace:
            dataset = self if inplace else self.copy_cubes_no_data()
        else:
            dataset = self.copy()

        for i, cube in enumerate(dataset.cubes):
            dataset.cubes[i] = (
                geo_subset(cube) if inplace else deepcopy(geo_subset(cube))
            )
        return dataset

    def regrid(
        self,
        new_latitudes=get_centres(np.linspace(-90, 90, 721)),
        new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
        **kwargs,
    ):
        """Replace stored cubes with regridded versions in-place."""
        # The time needed for this check is only on the order of ms.
        if all(lat_lon_match(cube, new_latitudes, new_longitudes) for cube in self):
            logger.info("No regridding needed for '{}'.".format(self.name))
        else:
            for cube_slice in self.single_cube_slices():
                self.cubes[cube_slice] = regrid_dataset(
                    self[cube_slice],
                    new_latitudes=new_latitudes,
                    new_longitudes=new_longitudes,
                    **kwargs,
                )

    @abstractmethod
    def get_monthly_data(self, start, end):
        """Return monthly cubes between two dates."""

    @staticmethod
    def date_order_check(start, end):
        if start is None and end is None:
            return
        if not all(
            getattr(date, required_type, None) is not None
            for date in (start, end)
            for required_type in ("year", "month")
        ):
            raise ValueError(
                "Both '{}' and '{}' need to define a year and month.".format(start, end)
            )
        days = [getattr(date, "day", 1) for date in (start, end)]
        days = [day if day is not None else 1 for day in days]

        assert datetime(start.year, start.month, days[0]) <= datetime(
            end.year, end.month, days[1]
        ), "End date must be greater than start date."

    def limit_months(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        """Discard time period outside the specified bounds.

        Crucially, this allows for regridding to take place much faster, as
        unused years/months are not considered.

        If the dataset consists of monthly data, the corresponding time
        period is selected, and the other times discarded.

        For yearly data, due to the need of interpolation, start/end
        dates are rounded down/up to the previous/next year respectively.

        """
        self.date_order_check(start, end)

        freq = self.frequency
        if freq in ("static", "monthly climatology"):
            logger.debug("Not limiting times, as data is static")
            return

        start = PartialDateTime(start.year, start.month)
        end = PartialDateTime(end.year, end.month)

        if freq == "yearly":
            start = PartialDateTime(start.year)
            if end.month != 1:
                end = PartialDateTime(end.year + 1)
        if freq not in ("monthly",):
            logger.warning("Encountered frequency:{:}".format(freq))
        self.cubes = self.cubes.extract(
            iris.Constraint(time=lambda t: end >= t.point >= start)
        )

    def select_monthly_from_monthly(
        self,
        start=PartialDateTime(2000, 1),
        end=PartialDateTime(2000, 12),
        inclusive_lower=True,
        inclusive_upper=True,
    ):
        """Select specified months from `self.cubes`.

        If only a single temporal coordinate matches the range specified by `start`
        and `end`, the resulting cube(s) will lack that dimension (ie. it is
        squeezed by Iris).

        """
        self.date_order_check(start, end)

        assert self.frequency == "monthly"

        lower_op = operator.ge if inclusive_lower else operator.gt
        upper_op = operator.le if inclusive_upper else operator.lt

        end = PartialDateTime(end.year, end.month)
        start = PartialDateTime(start.year, start.month)

        def constraint_func(t):
            return lower_op(t, start) and upper_op(t, end)

        return self.cubes.extract(
            iris.Constraint(time=lambda t: constraint_func(t.point))
        )

    def broadcast_static_data(self, start, end):
        """Broadcast every cube in 'self.cubes' to monthly intervals.

        Daily information is ignored (truncated, ie. days are assumed to be
        1).

        Limits are inclusive.

        """
        self.date_order_check(start, end)

        datetimes = [datetime(start.year, start.month, 1)]
        while datetimes[-1] != PartialDateTime(end.year, end.month):
            datetimes.append(datetimes[-1] + relativedelta(months=+1))

        calendar = "gregorian"
        time_unit_str = "days since 1970-01-01 00:00:00"
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)
        time_coord = iris.coords.DimCoord(
            cf_units.date2num(datetimes, time_unit_str, calendar),
            standard_name="time",
            units=time_unit,
        )

        new_cubes = iris.cube.CubeList()
        for cube in self.cubes:
            new_data = np.ma.vstack([cube.data[np.newaxis] for i in datetimes])
            coords = [
                (time_coord, 0),
                (cube.coord("latitude"), 1),
                (cube.coord("longitude"), 2),
            ]
            new_cubes.append(iris.cube.Cube(new_data, dim_coords_and_dims=coords))
            new_cubes[-1].metadata = cube.metadata

        return new_cubes

    def interpolate_yearly_data(self, start, end):
        """Linear interpolation onto the target months.

        Daily information is ignored (truncated, ie. days are assumed to be 1).

        Limits are inclusive.

        """
        self.date_order_check(start, end)

        time_unit = cf_units.Unit(self.time_unit_str, calendar=self.calendar)

        datetimes = [datetime(start.year, start.month, 1)]
        while datetimes[-1] != PartialDateTime(end.year, end.month):
            datetimes.append(datetimes[-1] + relativedelta(months=+1))

        time = iris.coords.DimCoord(
            cf_units.date2num(datetimes, self.time_unit_str, calendar=self.calendar),
            standard_name="time",
            units=time_unit,
        )

        interp_cubes = iris.cube.CubeList()
        for i in range(time.points.size):
            interp_points = [("time", time[i].points)]
            interp_cubes.extend(
                iris.cube.CubeList(
                    [
                        cube.interpolate(interp_points, iris.analysis.Linear())
                        for cube in self.cubes
                    ]
                )
            )

        final_cubelist = interp_cubes.concatenate()

        # Ignore extrema resulting from the interpolation (e.g. Iris linear
        # interpolation suffers from this problem sometimes).
        for cube in final_cubelist:
            original_data = self.cubes.extract_cube(
                iris.Constraint(name=cube.name())
            ).data
            omin, omax = np.min(original_data), np.max(original_data)
            # Augment the original mask.
            cube.data.mask = (
                cube.data.mask | (cube.data.data < omin) | (cube.data.data > omax)
            )

        return final_cubelist

    def single_cube_slices(self):
        """Get slices to select new datasets containing each variable one at a time.

        Using these slices in conjunction with cached method calls caches computations
        for each variable instead of one specific combination of variables, therefore
        making the caching much more flexible.

        """
        slices = []
        for index in range(len(self)):
            slices.append(slice(index, index + 1))
        return slices

    def get_mean_dataset(self):
        """Return new `Dataset` containing mean cubes between two dates.

        Note:
            Returned cubes may contain lazy data.

        """
        logger.info(f"Getting mean for '{self}'.")
        mean_cubes = iris.cube.CubeList()
        for cube_slice in self.single_cube_slices():
            mean_cubes.extend(get_dataset_mean_cubes(self[cube_slice]))

        mean_dataset = self.copy()
        mean_dataset.cubes = mean_cubes
        logger.debug(f"Finished getting mean for '{self}'.")
        return mean_dataset

    def get_monthly_dataset(self, start, end):
        """Return new `Dataset` containing monthly cubes between two dates.

        Note:
            Returned cubes will not contain lazy data, and this operation will realise
            all selected data!

        """
        logger.info(f"Getting monthly cubes for '{self}'.")
        monthly_cubes = iris.cube.CubeList()
        for cube_slice in self.single_cube_slices():
            monthly_cubes.extend(
                get_dataset_monthly_cubes(self[cube_slice], start, end)
            )

        monthly_dataset = self.copy()
        monthly_dataset.cubes = monthly_cubes
        logger.debug(f"Finished getting monthly cubes for '{self}'.")
        return monthly_dataset

    def get_climatology_dataset(self, start, end):
        """Return new `Dataset` containing monthly climatology cubes between two dates.

        Note:
            Returned cubes may contain lazy data.

        """
        logger.info(f"Getting monthly climatology for '{self}'.")
        climatology_cubes = iris.cube.CubeList()
        for cube_slice in self.single_cube_slices():
            climatology_cubes.extend(
                get_dataset_climatology_cubes(self[cube_slice], start, end)
            )

        climatology_dataset = self.copy()
        climatology_dataset.cubes = climatology_cubes
        logger.debug(f"Finished getting monthly climatology for '{self}'.")
        return climatology_dataset

    def get_mean_climatology_monthly_dataset(self, start, end):
        """Return new `Dataset` instances containing processed data.

        The output will contain the output of the `get_mean_dataset`,
        `get_climatology_dataset`, and `get_monthly_dataset` functions, except that it
        is slightly more efficient as fewer cache-retrieval operations need to be
        carried out.

        """
        mean_dataset = self.get_mean_dataset()

        logger.info(f"Getting monthly cubes for '{self}'.")
        monthly_cubes = iris.cube.CubeList()
        for cube_slice in self.single_cube_slices():
            monthly_cubes.extend(
                get_dataset_monthly_cubes(self[cube_slice], start, end)
            )

        monthly_dataset = self.copy()
        monthly_dataset.cubes = monthly_cubes
        logger.debug(f"Finished getting monthly cubes for '{self}'.")

        logger.info(f"Getting monthly climatology for '{self}'.")
        climatology_cubes = iris.cube.CubeList()
        # TODO: Implement this `optional_dataset` parameter which would allow passing
        # in the generated monthly cubes without having the function read it from the
        # cache.
        for cube_slice in self.single_cube_slices():
            climatology_cubes.extend(
                get_dataset_climatology_cubes(
                    self[cube_slice],
                    start,
                    end,
                    # TODO:
                    # optional_dataset=monthly_dataset[cube_slice],
                )
            )

        climatology_dataset = self.copy()
        climatology_dataset.cubes = climatology_cubes
        logger.debug(f"Finished getting monthly climatology for '{self}'.")
        return mean_dataset, climatology_dataset, monthly_dataset

    def get_observed_mask(self, thres=0.8, frequency=None):
        """Calculate a mask from the observed area and a minimum threshold.

        Args:
            thres (float): Minimum observed area threshold in [0, 1].
            frequency (str or None): If None, use the native temporal frequency. If
                "monthly", average observed fraction to monthly data before applying
                the threshold.

        Returns:
            iris.cube.Cube: Cube containing the Boolean mask.

        Raises:
            ObservedAreaError: If the `_observed_area` attribute is not defined, or
                the cube specified therein does not match one of the supported units
                (1,).

        """
        assert (
            0 <= thres <= 1
        ), f"Threshold needs to be in range [0, 1], but got '{thres}'."
        assert frequency in (None, "monthly"), (
            "Expected frequency to be one of 'None' and 'monthly', but got "
            f"'{frequency}'."
        )
        if not hasattr(self, "_observed_area"):
            raise ObservedAreaError(
                f"The dataset {self} does not specify the information necessary to "
                "determine the observed area mask."
            )

        if self._observed_area["name"] in self.variable_names("raw"):
            # If the cube containing the observed fraction is still present.
            target_dataset = self.copy()
        else:
            # Otherwise recreate all cubes.
            logger.warning(f"Recreating original cubes for {self}.")
            target_dataset = type(self)()

        target_dataset.cubes[:] = [
            deepcopy(target_dataset[self._observed_area["name"]])
        ]

        # Implement unit conversions here if needed.
        if target_dataset.cube.units != cf_units.Unit(1):
            raise ObservedAreaError(
                "Unsupported observed area unit '{self._observed_area['unit']}'."
            )

        if frequency == "monthly" and target_dataset.frequency != "monthly":
            logger.info("Converting mask to monthly data.")
            target_dataset = target_dataset.get_monthly_dataset(
                target_dataset.min_time, target_dataset.max_time
            )

        target_cube = target_dataset.cube
        observed_mask = target_cube.data.data < thres

        # Exchange data with the original (perhaps averaged) Cube for consistency.
        target_cube.data = observed_mask
        target_cube.units = cf_units.Unit("1")
        return target_cube

    @classmethod
    def get_obs_masked_dataset(cls, mask_vars, thres=0.8, ndigits=3, cached_only=False):
        """Create a new dataset based on masking of certain variables.

        The mask will be based on the observed area and the given threshold.

        Args:
            mask_vars ([iterable of] str): Variable(s) to mask using the observed area
                mask.
            thres (float): Minimum observed area threshold in [0, 1].
            ndigits (int): Number of digits to round `thres` to.
            cached_only (bool): If True, only load cached data. Otherwise return None.

        Returns:
            Instance of `cls` subclass or None: The name of this class will reflect the masked
                variables and the applied threshold.

        """
        rounded_thres = round(thres, ndigits)
        assert np.isclose(thres - rounded_thres, 0), (
            "Supplied threshold has too much precision. Either decrease precision or "
            "increase `ndigits`."
        )

        if isinstance(mask_vars, str):
            mask_vars = (mask_vars,)

        # Map given names to raw names if needed.
        raw_mask_vars = [
            cls._get_raw_variable_names()
            .get(name, name)
            .replace(" ", "_")
            .replace("-", "_")
            .replace("__", "_")
            .strip("_")
            for name in mask_vars
        ]

        name_mask_vars = [
            raw_name.replace(" ", "_").replace("-", "_").replace("__", "_").strip("_")
            for raw_name in raw_mask_vars
        ]

        format_str = "_thres_{rounded_thres:0." + str(ndigits) + "f}"
        pretty_format_str = "Thres {rounded_thres:0." + str(ndigits) + "f}"

        new_name = (
            cls.__name__
            + f"_{'__'.join(name_mask_vars)}_"
            + format_str.format(rounded_thres=rounded_thres)
        )

        # Initialise new Dataset instance.
        new_pretty_dataset_name = (
            cls._pretty
            + f" {' '.join(raw_mask_vars)} "
            + pretty_format_str.format(rounded_thres=rounded_thres)
        )

        # Intercept the cache writing operation in order to modify the cubes with the
        # observation mask before they get written to the cache. This will then also
        # affect subsequent retrievals of the cache.
        def new_cache_func(self):
            # Apply the mask. At this point the `cubes` attribute has already been
            # populated.

            # Retrieve the observation mask at the dataset-native frequency.
            obs_mask = cls().get_observed_mask(thres=rounded_thres)

            # Apply the mask to the cubes as set out in `mask_vars`.
            for var in raw_mask_vars:
                self[var].data.mask |= obs_mask.data

            # Call the original cache function to actually store the modified CubeList.
            cls.write_cache(self)

        masked_dataset_class = type(
            new_name,
            (cls,),
            {"_pretty": new_pretty_dataset_name, "write_cache": new_cache_func},
        )

        if cached_only and not masked_dataset_class.read_data(
            masked_dataset_class._get_cache_filename()
        ):
            return

        return masked_dataset_class()

    def get_temporally_shifted_dataset(self, months=0, deep=False):
        """Derive a new dataset with shifted temporal cubes.

        The definition of the sign of the shift is motivated by the investigation of
        pre-seasonal vegetation effects. Thus, `months=-1` shifts the data from
        January to February. Following this, an unshifted dataset's data from February
        could be compared to the shifted dataset's data from January by simply
        selecting the month February for both datasets.

        Args:
            months (int): Number of months to shift the "time" coordinates by.
            deep (bool): If True, copy the underlying data when creating the new
                dataset.

        Returns:
            An instance of a subclass of `type(self)` containing the shifted cubes.

        Raises:
            TypeError: If `months` is not an integer.
            UnexpectedCoordinateError: If any cube has multiple temporal coordinates.
            UnexpectedCoordinateError: If any cube has a temporal coordinate that
                isn't placed along the first axis.

        """
        if not isinstance(months, (int, np.integer)):
            raise TypeError(
                f"`months` should be an integer. Got type '{type(months)}'."
            )

        # Check temporal coordinates.
        for cube in self:
            # Cubes without a time coordinate are simply skipped.
            if not cube.coords("time"):
                continue
            # Ensure the time coordinate is first.
            if cube.coord_dims(cube.coord("time"))[0] != 0:
                raise UnexpectedCoordinateError(
                    "Temporal coordinate should correspond to the first axis."
                )
            if len(cube.coords(dimensions=0)) > 1:
                raise UnexpectedCoordinateError(
                    "There should only be a single temporal coordinate."
                )

        if deep:
            # Copy everything.
            orig_inst = self.copy(deep=True)
        else:
            # Copy everything but the underlying data, still making a copy of `self`,
            # `self.cubes` and the non-data attributes of every cube therein.
            orig_inst = self.copy_cubes_no_data()

        if not months:
            # No shift to be carried out - return instance of original class.
            return orig_inst

        if not orig_inst.frequency == "monthly":
            orig_inst = orig_inst.get_monthly_dataset(
                orig_inst.min_time, orig_inst.max_time
            )

        shift_dir = "plus" if months > 0 else "minus"

        def cube_name_mod_func(s):
            return s + f" {months} Month"

        # Handle each cube different, since each cube may have unique time coordinates
        # (different bands for example).
        for cube in orig_inst:
            if not cube.coords("time"):
                continue
            time_coord = cube.coord("time")
            time_coord.bounds = None
            shifted_dates = [
                ensure_datetime(time_coord.cell(i).point) - relativedelta(months=months)
                for i in range(len(time_coord.points))
            ]

            time_unit_str = time_coord.units.name
            time_unit_cal = time_coord.units.calendar

            num_shifted_dates = [
                cf_units.date2num(shifted_date, time_unit_str, time_unit_cal)
                for shifted_date in shifted_dates
            ]

            time_coord.points = num_shifted_dates

            cube.long_name = cube_name_mod_func(cube.name())
            cube.standard_name = None
            cube.var_name = None

        # Instantiate new dataset instance. This will lack any instantiation, which
        # must be replicated by manually assigning to the cubes attribute below.
        new_inst = type(
            self.name + f"__{shift_dir}_{abs(months)}_month",
            (type(self),),
            {
                "__init__": lambda self: None,
                "_pretty": self.pretty + f" {months} Month",
                "pretty_variable_names": dict(
                    (cube_name_mod_func(raw), cube_name_mod_func(pretty))
                    for raw, pretty in type(self).pretty_variable_names.items()
                ),
            },
        )()
        new_inst.cubes = orig_inst.cubes

        return new_inst

    def get_temporally_interpolated_dataset(
        self, target_timespan, n_months, mask=None, threshold=0, verbose=False
    ):
        """Temporal NN interpolation of missing data.

        Args:
            target_timespan (tuple of datetime): Start and end datetimes between which
                to interpolate.
            n_months (int): Number of months to look forwards and backwards to find
                valid data. The total number of months selected for each interpolation
                step will be `2 * n_months + 1` (including the current timestep).
            mask ((m, n) array of bool): Mask that is True where interpolation should
                occur. If None, interpolation is not carried out over water and south
                of 60 degrees latitude.
            threshold (float): Threshold in [0, 1] denoting the minimum fraction (not
                inclusive) of missing data in `source_masks` for which to carry out
                interpolation. Larger thresholds can be used to restrict interpolation
                geographically to regions with poor data availability.
            verbose (bool or int): If True, show progress bars. Giving 0 is equivalent
                to giving False (i.e. no progress bars), giving 1 shows a progress bar
                for the individual months, while giving 2 shows an additional progress
                bar for the individual samples within each month.

        Returns:
            An instance of a subclass of `type(self)` containing the interpolated cubes.

        Raises:
            ValueError: If `n_months` is not an integer.
            RuntimeError: If there was insufficient data to satisfy either the target or
                the source time period.

        """

        def cube_name_mod_func(s):
            return s + f" {n_months}NN"

        orig_inst = self.copy_cubes_no_data()
        interp_cubes = iris.cube.CubeList()
        # Handle each cube different, since each cube may have unique time coordinates
        # (different bands for example).
        for cube_slice in orig_inst.single_cube_slices():
            if not orig_inst[cube_slice].cube.coords("time"):
                continue
            cube = temporal_nn(
                orig_inst[cube_slice],
                target_timespan,
                n_months,
                mask=mask,
                threshold=threshold,
                verbose=verbose,
            )[0]

            cube.long_name = cube_name_mod_func(cube.name())
            cube.standard_name = None
            cube.var_name = None

            interp_cubes.append(cube)

        # Instantiate new dataset instance. This will lack any instantiation, which
        # must be replicated by manually assigning to the cubes attribute below.
        new_inst = type(
            self.name + f"__{n_months}NN",
            (type(self),),
            {
                "__init__": lambda self: None,
                "_pretty": cube_name_mod_func(self.pretty),
                "pretty_variable_names": dict(
                    (cube_name_mod_func(raw), cube_name_mod_func(pretty))
                    for raw, pretty in type(self).pretty_variable_names.items()
                ),
            },
        )()
        new_inst.cubes = interp_cubes

        return new_inst

    def get_persistent_season_trend_dataset(
        self, target_timespan=None, persistent_perc=50, k=4, verbose=False, ncpus=None
    ):
        """Interpolation of missing data using minimum values and season-trend model.

        First, persistent gaps are filled using the minimum value observed at that
        location. This is followed by the fitting of a season-trend model to fill the
        remaining missing values.

        Args:
            target_timespan (tuple of datetime or None): Start and end datetimes
                between which to interpolate. If None, the current temporal range of
                data will be used.
            persistent_perc (int in [0, 100]): Percentage of data that needs to be
                missing for a given month at a given location for the month to be
                considered affected by a persistent gap.
            k (int): Number of harmonic terms used in the season-trend model.
            verbose (bool): If True, show progress bars.
            ncpus (int or None): Number of processes to use for the season-trend model
                fitting. If None, `wildfires.qstat.get_ncpus()` will be used.

        Returns:
            An instance of a subclass of `type(self)` containing the interpolated cubes.

        Raises:
            ValueError: If `persistent_perc` is not an integer.
            ValueError: If `k` is not an integer.

        """

        def cube_name_mod_func(s):
            return s + f" {persistent_perc}P {k}k"

        if ncpus is None:
            ncpus = get_ncpus()

        orig_inst = self.copy_cubes_no_data()
        interp_cubes = iris.cube.CubeList()
        # Handle each cube different, since each cube may have unique time coordinates
        # (different bands for example).
        for cube_slice in orig_inst.single_cube_slices():
            if not orig_inst[cube_slice].cube.coords("time"):
                continue
            cube = persistent_season_trend_fill(
                orig_inst[cube_slice],
                target_timespan=target_timespan,
                persistent_perc=persistent_perc,
                k=k,
                verbose=verbose,
                ncpus=ncpus,
            )[0]

            cube.long_name = cube_name_mod_func(cube.name())
            cube.standard_name = None
            cube.var_name = None

            interp_cubes.append(cube)

        # Instantiate new dataset instance. This will lack any instantiation, which
        # must be replicated by manually assigning to the cubes attribute below.
        new_inst = type(
            self.name + f"__{persistent_perc}P_{k}k",
            (type(self),),
            {
                "__init__": lambda self: None,
                "_pretty": cube_name_mod_func(self.pretty),
                "pretty_variable_names": dict(
                    (cube_name_mod_func(raw), cube_name_mod_func(pretty))
                    for raw, pretty in type(self).pretty_variable_names.items()
                ),
            },
        )()
        new_inst.cubes = interp_cubes

        return new_inst


class MonthlyDataset(Dataset):
    """A `Dataset` containing monthly data."""

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class AvitabileAGB(Dataset):
    _pretty = "Avitabile AGB"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "Avitabile_AGB")
        self.cubes = iris.cube.CubeList(
            [iris.load_cube(os.path.join(self.dir, "Avitabile_AGB_Map_0d25.nc"))]
        )

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.broadcast_static_data(start, end)


class AvitabileThurnerAGB(Dataset):
    _pretty = "Avitabile Thurner AGB"
    pretty_variable_names = {"AGBtree": "AGB Tree"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "AvitabileThurner-merged_AGB")
        self.cubes = iris.cube.CubeList(
            [
                iris.load_cube(
                    os.path.join(self.dir, "Avi2015-Thu2014-merged_AGBtree.nc")
                )
            ]
        )

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.broadcast_static_data(start, end)


class CarvalhaisGPP(Dataset):
    _pretty = "Carvalhais GPP"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "Carvalhais_VegC-TotalC-Tau")
        raw_cubes = iris.cube.CubeList(
            [iris.load_cube(os.path.join(self.dir, "Carvalhais.gpp_50.360.720.1.nc"))]
        )
        # There is only one time coordinate, and its value is of no relevance.
        # Therefore, remove this coordinate.
        raw_cubes[0] = raw_cubes[0][0]
        raw_cubes[0].remove_coord("time")

        self.cubes = raw_cubes

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.broadcast_static_data(start, end)


class CCI_BurnedArea_MERIS_4_1(Dataset):
    _pretty = "CCI MERIS 4.1"
    pretty_variable_names = {"burned_area": "CCI MERIS BA"}
    _special_coord_cubes = {
        "vegetation class name": ["vegetation_class"],
        "burned area in vegetation class": [
            "time",
            "vegetation_class",
            "latitude",
            "longitude",
        ],
    }
    _observed_area = {"name": "fraction of observed area"}

    def __init__(self):
        # Manually input directory name here to maintain this directory for subclasses.
        self.dir = os.path.join(DATA_DIR, "CCI_BurnedArea_MERIS_4_1")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        cubes = iris.cube.CubeList()
        for f in tqdm(
            glob.glob(os.path.join(self.dir, "**", "*.nc"), recursive=True),
            desc="Loading cubes",
        ):
            cubes.extend(iris.load(f))

        named_cubes = dict(
            [
                (var_name, cubes.extract(iris.Constraint(var_name)))
                for var_name in set([cube.name() for cube in cubes])
            ]
        )

        for var_name, var_cubes in tqdm(
            named_cubes.items(), desc="Homogenising cube attributes"
        ):
            # TODO: Fuse some of the discarded attributes, like the time coverage.
            homogenise_cube_attributes(var_cubes)
            var_cube = var_cubes[0]
            assert all(
                var_cube.is_compatible(var_cubes[i]) for i in range(1, len(var_cubes))
            ), "Should be able to concatenate cubes now."

            if var_name == "vegetation class name":
                # All cubes are the same (except for isolated metadata, like timing
                # information) so we only deal with one cube.

                # Convert '|S1' dtype to 'u1' ('uint8') dtype to avoid errors during storage.
                # Replace b'' placeholder values with b' ' to enable usage of `ord'.
                var_cube.data.data[var_cube.data.mask] = b" "

                int_veg_data = np.asarray(
                    np.vectorize(ord)(var_cube.data.data), dtype="u1"
                )
                var_cube.data = np.ma.MaskedArray(
                    int_veg_data, mask=var_cube.data.mask, dtype="u1", fill_value=32
                )
                # NOTE: Figure out why the masked data, the mask itself, and the fill
                # value are modified when saving the cube with the data created above.

                named_cubes[var_name] = iris.cube.CubeList([var_cube])
            else:
                # The time bounds seem to be wrong in the original data, so remove them.
                for cube in var_cubes:
                    cube.coord("time").bounds = None

        raw_cubes = iris.cube.CubeList(
            [var_cubes.concatenate_cube() for var_cubes in named_cubes.values()]
        )

        for i, cube in enumerate(tqdm(raw_cubes, desc="Normalising cubes")):
            if cube.name() in [
                "burned_area",
                "burned area in vegetation class",
                "standard error of the estimation of burned area",
            ]:
                # Normalise using the grid cell areas
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=(("Using DEFAULT_SPHERICAL_EARTH_RADIUS."))
                    )
                    raw_cubes[i].data /= iris.analysis.cartography.area_weights(
                        raw_cubes[i]
                    )
                raw_cubes[i].units = cf_units.Unit(1)

            # Rewrite coordinate `long_name' values to conform to netCDF variable name
            # standards to ensure compatibility with the coordinate `var_name'
            # requirements.
            if cube.name() == "burned area in vegetation class":
                cube.coords()[1].long_name = "vegetation_class"
            elif cube.name() == "vegetation class name":
                cube.coords()[0].long_name = "vegetation_class"

        self.cubes = raw_cubes
        self.write_cache()

    @property
    def vegetation_class_names(self):
        """Retrieve the vegetation class names."""
        # Make the vegetation names persist even if the corresponding cube is removed.
        if hasattr(self, "_cached_vegetation_class_names"):
            return self._cached_vegetation_class_names

        if "vegetation class name" in self.variable_names("raw"):
            # If the cube containing the names is still present.
            target_dataset = self
        else:
            # Otherwise recreate all cubes and extract the needed data.
            target_dataset = type(self)()

        vegetation_cube = target_dataset["vegetation class name"]
        # Remove artefacts of saving the cube. See note above.
        vegetation_class_names = [
            "".join(class_name_data).strip().strip(chr(255))
            for class_name_data in np.vectorize(chr)(vegetation_cube.data.data)
        ]

        self._cached_vegetation_class_names = vegetation_class_names
        return vegetation_class_names

    def get_monthly_data(
        self,
        start=PartialDateTime(2000, 1),
        end=PartialDateTime(2000, 12),
        inclusive_lower=True,
        inclusive_upper=True,
    ):
        """Transform the data from two samples a month to having just one."""
        self.date_order_check(start, end)

        lower_op = operator.ge if inclusive_lower else operator.gt
        upper_op = operator.le if inclusive_upper else operator.lt

        end = PartialDateTime(end.year, end.month)
        start = PartialDateTime(start.year, start.month)

        def constraint_func(t):
            return lower_op(t, start) and upper_op(t, end)

        monthly_cubes = iris.cube.CubeList()
        for cube in self.cubes.extract(
            iris.Constraint(time=lambda t: constraint_func(t.point))
        ):
            try:
                iris.coord_categorisation.add_month_number(cube, "time")
                iris.coord_categorisation.add_year(cube, "time")
                monthly_cubes.append(
                    cube.aggregated_by(["month_number", "year"], iris.analysis.MEAN)
                )
            except iris.exceptions.CoordinateNotFoundError:
                monthly_cubes.append(cube)

        return monthly_cubes


class CCI_BurnedArea_MODIS_5_1(MonthlyDataset):
    _pretty = "CCI MODIS 5.1"
    pretty_variable_names = {"burned_area": "CCI MODIS BA"}
    _special_coord_cubes = {
        "vegetation class name": ["vegetation_class"],
        "burned area in vegetation class": [
            "time",
            "vegetation_class",
            "latitude",
            "longitude",
        ],
    }

    def __init__(self):
        # Manually input directory name here to maintain this directory for subclasses.
        self.dir = os.path.join(DATA_DIR, "CCI_BurnedArea_MODIS_5_1")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        cubes = iris.cube.CubeList()
        for f in tqdm(
            glob.glob(os.path.join(self.dir, "**", "*.nc"), recursive=True),
            desc="Loading cubes",
        ):
            cubes.extend(iris.load(f))

        named_cubes = dict(
            [
                (var_name, cubes.extract(iris.Constraint(var_name)))
                for var_name in set([cube.name() for cube in cubes])
            ]
        )

        for var_name, var_cubes in tqdm(
            named_cubes.items(), desc="Homogenising cube attributes"
        ):
            # TODO: Fuse some of the discarded attributes, like the time coverage.
            homogenise_cube_attributes(var_cubes)
            var_cube = var_cubes[0]
            assert all(
                var_cube.is_compatible(var_cubes[i]) for i in range(1, len(var_cubes))
            ), "Should be able to concatenate cubes now."

            if var_name == "vegetation class name":
                # All cubes are the same (except for isolated metadata, like timing
                # information) so we only deal with one cube.

                # Convert '|S1' dtype to 'u1' ('uint8') dtype to avoid errors during storage.
                # Replace b'' placeholder values with b' ' to enable usage of `ord'.
                var_cube.data.data[var_cube.data.mask] = b" "

                int_veg_data = np.asarray(
                    np.vectorize(ord)(var_cube.data.data), dtype="u1"
                )
                var_cube.data = np.ma.MaskedArray(
                    int_veg_data, mask=var_cube.data.mask, dtype="u1", fill_value=32
                )
                # NOTE: Figure out why the masked data, the mask itself, and the fill
                # value are modified when saving the cube with the data created above.

                named_cubes[var_name] = iris.cube.CubeList([var_cube])

        raw_cubes = iris.cube.CubeList(
            [var_cubes.concatenate_cube() for var_cubes in named_cubes.values()]
        )

        for i, cube in enumerate(tqdm(raw_cubes, desc="Normalising cubes")):
            if cube.name() in [
                "burned_area",
                "burned area in vegetation class",
                "standard error of the estimation of burned area",
            ]:
                # Normalise using the grid cell areas
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=(("Using DEFAULT_SPHERICAL_EARTH_RADIUS."))
                    )
                    raw_cubes[i].data /= iris.analysis.cartography.area_weights(
                        raw_cubes[i]
                    )
                raw_cubes[i].units = cf_units.Unit(1)

            # Rewrite coordinate `long_name' values to conform to netCDF variable name
            # standards to ensure compatibility with the coordinate `var_name'
            # requirements.
            if cube.name() == "burned area in vegetation class":
                cube.coords()[1].long_name = "vegetation_class"
            elif cube.name() == "vegetation class name":
                cube.coords()[0].long_name = "vegetation_class"

        self.cubes = raw_cubes
        self.write_cache()

    @property
    def vegetation_class_names(self):
        vegetation_cube = self["vegetation class name"]
        # Remove artefacts of saving the cube. See note above.
        vegetation_class_names = [
            "".join(class_name_data).strip().strip(chr(255))
            for class_name_data in np.vectorize(chr)(vegetation_cube.data.data)
        ]
        return vegetation_class_names


class CHELSA(MonthlyDataset):
    """For primary analysis, it is advisable to use hpc
    (cx1_scipts/run_chelsa_script.sh) in order to process the tif files
    into nc files as a series of jobs, which would take an incredibly long
    time otherwise (on the order of days).

    Once that script has been run, the resulting nc files can be used to
    easily construct a large iris Cube containing all the data.

    """

    _pretty = "CHELSA"
    pretty_variable_names = {
        "maximum temperature": "Max Temp",
        "minimum temperature": "Min Temp",
        "mean temperature": "Mean Temp",
        "monthly precipitation": "Precipitation",
    }

    def __init__(self, process_slice=slice(None)):
        """Initialise the cubes.

        Args:
            process_slice (slice): Used to limit the loading/processing of
                raw data .tif data files. Slices resulting in single
                elements (eg. slice(i, i+1)) can be provided with i being
                the PBS array job index (for example) to quickly generate
                all the required .nc files from the .tif files using array
                jobs on the hpc.

        """
        self.dir = os.path.join(DATA_DIR, "CHELSA")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        files = glob.glob(os.path.join(self.dir, "**", "*.tif"), recursive=True)
        files.sort()

        mapping = {
            "prec": {
                "scale": 1,
                "unit": cf_units.Unit("mm/month"),
                "long_name": "monthly precipitation",
            },
            "tmax": {
                "scale": 0.1,
                "unit": cf_units.Unit("degrees Celsius"),
                "long_name": "maximum temperature",
            },
            "tmean": {
                "scale": 0.1,
                "unit": cf_units.Unit("degrees Celsius"),
                "long_name": "mean temperature",
            },
            "tmin": {
                "scale": 0.1,
                "unit": cf_units.Unit("degrees Celsius"),
                "long_name": "minimum temperature",
            },
        }

        year_pattern = re.compile(r"_(\d{4})_")
        month_pattern = re.compile(r"_(\d{2})_")

        time_unit_str = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        commit_hashes = set()
        cube_list = iris.cube.CubeList()

        def update_hashes(commit_hash):
            commit_hashes.update([commit_hash])
            # TODO: Need to reinstate this constraint!!!!
            """
            assert len(commit_hashes) == 1, (
                    "All loaded data should be from the same commit.")
            """

        for f in files[process_slice]:
            # If this file has been regridded already and saved as a NetCDF
            # file, then do not redo this.
            nc_file = f.replace(".tif", ".nc")
            try:
                cubes = self.read_data(nc_file)
            except Exception:
                # Try again, removing a potentially corrupt file
                # beforehand.
                logger.exception("Read failed, recreating:'{:}'".format(nc_file))
                cubes = None
                try:
                    os.remove(nc_file)
                except Exception:
                    logger.exception("File did not exist:'{:}'".format(nc_file))

            if cubes:
                update_hashes(cubes[0].attributes["commit"])
                cube_list.extend(cubes)
                continue

            try:
                with rasterio.open(f) as dataset:
                    pass
            except rasterio.RasterioIOError:
                logger.exception("Corrupted file.")
                # Try to download file again.
                url = f.replace(
                    os.path.join(DATA_DIR, "CHELSA"), "https://www.wsl.ch/lud/chelsa"
                )

                command = "curl --connect-timeout 20 -L -o {:} {:}".format(f, url)
                logger.debug("Executing:{:}".format(command))
                os.system(command)

            with rasterio.open(f) as dataset:
                # NOTE: Since data is are stored as unsigned 16 bit
                # integers, with temperature (in degrees Celsius) scaled by
                # a factor x10, space can be saved by saving data in
                # float16 format.
                variable_key = os.path.split(os.path.split(f)[0])[1]
                assert dataset.count == 1, "There should only be one band."
                data = dataset.read(1).astype("float16")
                data = np.ma.MaskedArray(
                    data * mapping[variable_key]["scale"],
                    np.isinf(data),
                    dtype=data.dtype,
                )

                latitudes = iris.coords.DimCoord(
                    get_centres(
                        np.linspace(
                            dataset.bounds.top,
                            dataset.bounds.bottom,
                            dataset.shape[0] + 1,
                        )
                    ),
                    standard_name="latitude",
                    units="degrees",
                )
                longitudes = iris.coords.DimCoord(
                    get_centres(
                        np.linspace(
                            dataset.bounds.left,
                            dataset.bounds.right,
                            dataset.shape[1] + 1,
                        )
                    ),
                    standard_name="longitude",
                    units="degrees",
                )

            grid_coords = [(latitudes, 0), (longitudes, 1)]

            split_f = os.path.split(f)[1]
            time_coord = iris.coords.DimCoord(
                cf_units.date2num(
                    datetime(
                        int(year_pattern.search(split_f).group(1)),
                        int(month_pattern.search(split_f).group(1)),
                        1,
                    ),
                    time_unit_str,
                    calendar,
                ),
                standard_name="time",
                units=time_unit,
            )

            cube = iris.cube.Cube(
                data,
                dim_coords_and_dims=grid_coords,
                units=mapping[variable_key]["unit"],
                var_name=variable_key,
                long_name=mapping[variable_key]["long_name"],
                aux_coords_and_dims=[(time_coord, None)],
            )

            # Regrid cubes to the same lat-lon grid.
            # TODO: change lat and lon limits and also the number of points!!
            # Always work in 0.25 degree steps? From the same starting point?
            regrid_cube = regrid(cube)

            # Need to save as float64 or float32, choose float64 for future
            # interoperability.
            regrid_cube.data = regrid_cube.data.astype("float64")
            commit_hash = self.save_data(regrid_cube, nc_file)

            # If None is returned, then the file already exists and is not
            # being overwritten, which should not happen, as we check for
            # the existence of the file above, loading the data in that
            # case.
            assert (
                commit_hash is not None
            ), "Data should have been loaded before, since the file exists."
            update_hashes(commit_hash)
            cube_list.append(regrid_cube)

        # TODO: TEMPORARY, in order to allow merging of data from different
        # commits!!
        for cube in cube_list:
            del cube.attributes["commit"]

        self.cubes = cube_list.merge()
        assert len(self.cubes) == 4, "There should be 4 variables."

        # If all the data has been processed, not just a subset.
        if process_slice == slice(None):
            self.write_cache()


class Copernicus_SWI(MonthlyDataset):
    """For primary analysis, it is advisable to use hpc
    (cx1_scipts/run_swi_script.sh) in order to process the daily nc files
    into monthly nc files as a series of jobs, which would take an
    incredibly long time and large amounts of RAM otherwise (on the order
    of days).

    Once that script has been run, the resulting nc files can be used to
    easily construct a large iris Cube containing all the desired monthly
    data.

    There are currently 147 available months of data, from 2007-01 to
    2019-03.

    """

    _pretty = "Copernicus SWI"
    pretty_variable_names = {"Soil Water Index with T=1": "SWI(1)"}

    def __init__(self, process_slice=slice(None)):
        """Initialise the cubes.

        Args:
            process_slice (slice): Used to limit the loading/processing of
                raw daily .nc files. Slices resulting in single elements
                (eg. slice(i, i+1)) will select a MONTH of data. For
                example, this can be done with i being the PBS array job
                index (for example) to quickly generate all the required
                monthly .nc files from the daily files using array jobs on
                the hpc.

        """
        self.dir = os.path.join(DATA_DIR, "Copernicus_SWI")
        logger.debug("Copernicus dir:{:}".format(self.dir))
        monthly_dir = os.path.join(self.dir, "monthly")
        logger.debug("Monthly dir:{:}".format(monthly_dir))

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        # The raw data is daily data, which has to be averaged to yield
        # monthly data.
        files = glob.glob(os.path.join(self.dir, "**", "*.nc"), recursive=True)

        daily_files = []
        monthly_files = []
        for f in files:
            if "monthly" in f:
                monthly_files.append(f)
            else:
                daily_files.append(f)

        logger.debug(
            "Found {:} monthly & {:} daily files".format(
                len(monthly_files), len(daily_files)
            )
        )

        # Get times from the filenames, instead of having to load the cubes
        # and look at the time coordinate that way.
        pattern = re.compile(r"(\d{4})(\d{2})(\d{2})")
        datetimes = [datetime(*map(int, pattern.search(f).groups())) for f in files]

        # Isolate the year and month of each file only, and only in the
        # times of the requested slice.
        year_months = sorted(
            list(set([datetime(dt.year, dt.month, 1) for dt in datetimes]))
        )[process_slice]

        start_year_month = year_months[0]
        end_year_month = year_months[-1] + relativedelta(months=+1)
        logger.debug(
            "Processing data from {:} to {:}".format(start_year_month, end_year_month)
        )

        selected_daily_files = []

        selected_monthly_files = []
        selected_monthly_intervals = []

        # Handle monthly files first, in order to eliminate double-counting
        # later on.
        for f, dt in zip(files, datetimes):
            if start_year_month <= dt < end_year_month:
                # Prevent loading monthly files into the daily file list
                # which will get processed into monthly data.
                #
                # Only ignore the 1 month interval which is associated with
                # each monthly file. If multiple intervals are found, they
                # will be merged later.
                if "monthly" in f:
                    selected_monthly_files.append(f)
                    selected_monthly_intervals.append(
                        [dt, dt + relativedelta(months=+1)]
                    )

        # Fuse the monthly intervals into easier-to-use contiguous
        # intervals.
        contiguous_monthly_intervals = join_adjacent_intervals(
            selected_monthly_intervals
        )

        logger.debug(
            "Contiguous monthly intervals:{:}".format(contiguous_monthly_intervals)
        )

        for f, dt in zip(files, datetimes):
            if start_year_month <= dt < end_year_month:
                monthly_data = False
                for interval in contiguous_monthly_intervals:
                    if interval[0] <= dt < interval[1]:
                        monthly_data = True

                if not monthly_data:
                    assert (
                        "monthly" not in f
                    ), "Monthly files should have been separated beforehand."
                    selected_daily_files.append(f)

        logger.debug(
            "Using {:} monthly & {:} daily files".format(
                len(selected_monthly_files), len(selected_daily_files)
            )
        )

        commit_hashes = set()
        monthly_cubes = iris.cube.CubeList()

        def update_hashes(commit_hash):
            commit_hashes.update([commit_hash])
            # TODO: Need to reinstate this constraint!!!!
            """
            assert len(commit_hashes) == 1, (
                    "All loaded data should be from the same commit.")
            """

        # Process the daily files here first, then combine with the already
        # processed monthly data later. Processing involves regridding to a
        # 0.25 degree resolution and averaging over months.
        if selected_daily_files:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "Skipping global attribute 'long_name': 'long_name' is "
                        "not a permitted attribute"
                    ),
                )
                daily_cubes = load_cubes(selected_daily_files)

            # Concatenate daily cubes into larger cubes with the same
            # information (but with longer time coordinates).
            raw_cubes = homogenise_cube_attributes(daily_cubes).concatenate()

            while raw_cubes:
                logger.debug("Regridding:{:}".format(repr(raw_cubes[0])))
                regridded_cube = regrid(raw_cubes.pop(0), scheme=iris.analysis.Linear())
                iris.coord_categorisation.add_month_number(regridded_cube, "time")
                iris.coord_categorisation.add_year(regridded_cube, "time")
                logger.debug("Averaging:{:}".format(repr(regridded_cube)))

                averaged_cube = regridded_cube.aggregated_by(
                    ["month_number", "year"], iris.analysis.MEAN
                )

                assert averaged_cube.core_data().shape[0] == 1, (
                    "There should be only 1 element in the time dimension "
                    "(ie. 1 month)."
                )

                monthly_cubes.append(averaged_cube[0])

                logger.debug(
                    "Remaining nr to regrid & average:{:}".format(len(raw_cubes))
                )

            # Save these monthly files separately.
            datetimes_to_save = []
            for cube in monthly_cubes:
                for i in range(len(cube.coord("time").points)):
                    datetimes_to_save.append(cube.coord("time").cell(i).point)
            datetimes_to_save = list(set(datetimes_to_save))

            for dt in datetimes_to_save:
                cubes = monthly_cubes.extract(
                    iris.Constraint(time=lambda t: dt == t.point)
                )

                commit_hash = self.save_data(
                    cubes,
                    os.path.join(
                        monthly_dir,
                        (
                            "c_gls_SWI_{:04d}{:02d}{:02d}_monthly"
                            "_GLOBE_ASCAT_V3.1.1.nc"
                        ).format(
                            # The day is always 1 for monthly files.
                            dt.year,
                            dt.month,
                            1,
                        ),
                    ),
                )

                # If None is returned, then the file already exists and is not
                # being overwritten, which should not happen, as we check for
                # the existence of the file above, loading the data in that
                # case.
                assert (
                    commit_hash is not None
                ), "Data should have been loaded before, since the file exists."
                update_hashes(commit_hash)

        if selected_monthly_files:
            monthly_cubes.extend(load_cubes(selected_monthly_files))

        # TODO: TEMPORARY, in order to allow merging of data from different
        # commits!!
        for cube in monthly_cubes:
            if "commit" in cube.attributes:
                del cube.attributes["commit"]

        logger.debug("Merging final cubes.")
        # TODO: Verify that this works as expected.
        merged_cubes = monthly_cubes.merge()
        self.cubes = iris.cube.CubeList(
            cube
            for cube in merged_cubes
            if cube.attributes["processing_mode"] == "Reprocessing"
        )

        logger.debug("Finished merging.")

        # If all the data has been processed, not just a subset.
        if process_slice == slice(None):
            logger.debug("Writing cache for entire timespan")
            self.write_cache()


class CRU(MonthlyDataset):
    _pretty = "CRU"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "CRU")
        # Ignore warning regarding cloud cover units - they are fixed below.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=("Ignoring netCDF variable 'cld' invalid units 'percentage'"),
            )

            # TODO: In order to use the 'stn' variable - with information
            # about the measurement stations, the files have to be handled
            # individually so that we can keep track of which stn cube
            # belongs to which data cube.
            raw_cubes = iris.load(os.path.join(self.dir, "*.nc"))

        # TODO: For now, remove the 'stn' cubes (see above).
        self.cubes = iris.cube.CubeList(
            [cube for cube in raw_cubes if cube.name() != "stn"]
        )

        # Fix units for cloud cover.
        for cube in self.cubes:
            if cube.name() == "cloud cover":
                cube.units = cf_units.Unit("percent")
                break

        # NOTE: Measurement times are listed as being in the middle of the
        # month, requiring no further intervention.


class ERA5_Temperature(MonthlyDataset):
    _pretty = "ERA5 Temperature"
    pretty_variable_names = {
        "Mean 2 metre temperature": "Mean Temp",
        "Min 2 metre temperature": "Min Temp",
        "Max 2 metre temperature": "Max Temp",
        "DTR 2 metre temperature": "Diurnal Temp Range",
    }

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ERA5", "temperature")
        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return
        files = sorted(
            glob.glob(
                os.path.join(self.dir, "**", "*_monthly_mean_min_max_dtr.nc"),
                recursive=True,
            )
        )
        if not files:
            logger.info("No processed files found. Downloading and processing now.")
            retrieval_processing(
                retrieve(
                    variable="2t",
                    start=PartialDateTime(1990, 1, 1),
                    end=PartialDateTime(2019, 1, 1),
                    target_dir=self.dir,
                ),
                processing_class=MonthlyMeanMinMaxDTRWorker,
                n_threads=10,
                soft_filesize_limit=700,
            )
            files = sorted(
                glob.glob(
                    os.path.join(self.dir, "**", "*_monthly_mean_min_max_dtr.nc"),
                    recursive=True,
                )
            )
            assert files

        self.cubes = homogenise_cube_attributes(load_cubes(files)).merge()
        self.write_cache()


class ERA5_TotalPrecipitation(MonthlyDataset):
    _pretty = "ERA5 Total Precipitation"
    pretty_variable_names = {"Total precipitation": "Precipitation"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ERA5", "tp")
        self.cubes = iris.cube.CubeList(
            [iris.load_cube(os.path.join(self.dir, "*.nc"))]
        )


class ERA5_DryDayPeriod(MonthlyDataset):
    _pretty = "ERA5 Dry Day Period"
    pretty_variable_names = {"dry_day_period": "Dry Day Period"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ERA5", "tp_daily")
        self.cubes = self.read_cache()
        if self.cubes:
            return

        # Sort so that time is increasing.
        filenames = sorted(
            glob.glob(os.path.join(self.dir, "**", "*_daily_mean.nc"), recursive=True)
        )

        logger.info("Constructing dry day period cube.")
        dry_day_period_cubes = iris.cube.CubeList()

        prev_dry_day_period = None
        prev_end = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Collapsing a non-contiguous coordinate. Metadata may not "
                    "be fully descriptive for 'time'."
                ),
            )
            for filename in tqdm(filenames):
                raw_cube = iris.load_cube(filename)
                n_days = raw_cube.shape[0]
                n_lats = raw_cube.shape[1]
                n_lons = raw_cube.shape[2]

                # The first time around only, create empty arrays. This will introduce
                # some negative bias for the first month(s), but this should be
                # negligible overall (especially since the first year is probably not
                # being used anyway).
                if prev_dry_day_period is None:
                    assert prev_end is None
                    prev_dry_day_period = np.zeros((n_lats, n_lons), dtype=np.int64)
                    prev_end = np.zeros((n_lats, n_lons), dtype=np.bool_)

                # Calculate dry days using metre per hour threshold, since the daily
                # data here is an average of the hourly total precipitation data.
                dry_days = raw_cube.data < M_PER_HR_THRES

                # Find contiguous blocks in the time dimension where dry_days is True.
                structure = np.zeros((3, 3, 3), dtype=np.int64)
                structure[:, 1, 1] = 1
                labelled = scipy.ndimage.label(dry_days, structure=structure)
                slices = scipy.ndimage.find_objects(labelled[0])

                dry_day_period = np.zeros((n_lats, n_lons), dtype=np.int64)
                beginning = np.zeros((n_lats, n_lons), dtype=np.bool_)
                end = np.zeros_like(beginning)

                for slice_object in slices:
                    time_slice = slice_object[0]
                    lat_slice = slice_object[1]
                    lon_slice = slice_object[2]
                    assert lat_slice.stop - lat_slice.start == 1
                    assert lon_slice.stop - lon_slice.start == 1

                    latitude = lat_slice.start
                    longitude = lon_slice.start

                    period_length = time_slice.stop - time_slice.start

                    if period_length > dry_day_period[latitude, longitude]:
                        dry_day_period[latitude, longitude] = period_length
                        if time_slice.start == 0:
                            beginning[latitude, longitude] = True
                        else:
                            beginning[latitude, longitude] = False
                        if time_slice.stop == n_days:
                            end[latitude, longitude] = True
                        else:
                            end[latitude, longitude] = False

                # Once the data for the current month has been processed, look at the
                # previous month to see if dry day periods may be joined up.
                overlap = prev_end & beginning
                dry_day_period[overlap] += prev_dry_day_period[overlap]

                # Prepare for the next month's analysis.
                prev_dry_day_period = dry_day_period
                prev_end = end

                # Create new Cube with the same latitudes and longitudes, and an
                # averaged time.
                coords = [
                    (raw_cube.coord("latitude"), 0),
                    (raw_cube.coord("longitude"), 1),
                ]

                # Modify the time coordinate such that it is recorded with
                # respect to a common date, as opposed to relative to the
                # beginning of the respective month as is the case for the
                # cube loaded above.

                # Take the new 'mean' time as the average of the first and last time.
                min_time = raw_cube.coord("time").cell(0).point
                max_time = raw_cube.coord("time").cell(-1).point
                centre_datetime = min_time + ((max_time - min_time) / 2)

                new_time = cf_units.date2num(
                    centre_datetime, self.time_unit_str, self.calendar
                )
                time_coord = iris.coords.DimCoord(
                    new_time, units=self.time_unit, standard_name="time"
                )

                dry_day_period_cube = iris.cube.Cube(
                    dry_day_period,
                    dim_coords_and_dims=coords,
                    units=cf_units.Unit("days"),
                    var_name="dry_day_period",
                    aux_coords_and_dims=[(time_coord, None)],
                )
                dry_day_period_cube.units = cf_units.Unit("days")

                dry_day_period_cubes.append(dry_day_period_cube)

        raw_cubes = iris.cube.CubeList([dry_day_period_cubes.merge_cube()])

        self.cubes = raw_cubes
        self.write_cache()


class ERA5_CAPEPrecip(MonthlyDataset):
    _pretty = "ERA5 Cape x Precip"
    pretty_variable_names = {"Product of CAPE and Precipitation": "CAPE x Precip"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ERA5", "CAPE_P")
        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return
        files = sorted(glob.glob(os.path.join(self.dir, "**", "*.nc"), recursive=True))
        raw_cubes = load_cubes(files)
        self.cubes = iris.cube.CubeList([raw_cubes.merge_cube()])
        self.write_cache()


class ESA_CCI_Fire(Dataset):
    _pretty = "ESA CCI Fire"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ESA-CCI-Fire_burnedarea")
        self.cubes = iris.cube.CubeList(
            [
                iris.load_cube(
                    os.path.join(
                        self.dir, "MODIS_cci.BA.2001.2016.1440.720.365days.sum.nc"
                    )
                )
            ]
        )
        self.time_unit_str = self.cubes[0].coord("time").units.name
        self.calendar = self.cubes[0].coord("time").units.calendar

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.interpolate_yearly_data(start, end)


class ESA_CCI_Landcover(Dataset):
    _pretty = "ESA Landcover"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ESA-CCI-LC_landcover", "0d25_landcover")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        filenames = glob.glob(os.path.join(self.dir, "*.nc"))
        filenames.sort()  # increasing years
        self.raw_cubes = iris.load(filenames)

        # To concatenate the cubes, take advantage of the fact that there
        # are 17 cubes per year, and then simply loop over the years,
        # joining the corresponding cubes into lists corresponding to their
        # variable.
        cube_lists = []
        for i in range(17):
            cube_lists.append(iris.cube.CubeList())

        n_years = len(self.raw_cubes) / 17
        assert np.isclose(n_years, int(n_years))
        n_years = int(n_years)

        years = range(1992, 2016)
        assert len(years) == n_years

        self.time_unit_str = "hours since 1970-01-01 00:00:00"
        self.calendar = "gregorian"
        time_unit = cf_units.Unit(self.time_unit_str, calendar=self.calendar)

        for i in range(n_years):
            time = iris.coords.DimCoord(
                [
                    cf_units.date2num(
                        datetime(years[i], 1, 1), self.time_unit_str, self.calendar
                    )
                ],
                standard_name="time",
                units=time_unit,
            )
            for j in range(17):
                cube = self.raw_cubes[(17 * i) + j]

                cube_coords = cube.coords()

                cube2 = iris.cube.Cube(cube.lazy_data().reshape(1, 720, 1440))
                cube2.attributes = cube.attributes
                cube2.long_name = cube.long_name
                cube2.name = cube.name
                cube2.standard_name = cube.standard_name
                cube2.units = cube.units
                cube2.var_name = cube.var_name

                for key in ["id", "tracking_id", "date_created"]:
                    del cube2.attributes[key]
                cube2.attributes["time_coverage_start"] = self.raw_cubes[0].attributes[
                    "time_coverage_start"
                ]
                cube2.attributes["time_coverage_end"] = self.raw_cubes[-1].attributes[
                    "time_coverage_end"
                ]

                cube2.add_dim_coord(time, 0)
                cube2.add_dim_coord(cube_coords[0], 1)
                cube2.add_dim_coord(cube_coords[1], 2)

                cube_lists[j].append(cube2)

        self.cubes = iris.cube.CubeList()
        for cube_list in cube_lists:
            self.cubes.append(cube_list.concatenate_cube())

        self.write_cache()

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.interpolate_yearly_data(start, end)


class Ext_ESA_CCI_Landcover_PFT(Dataset):
    """Extended ESA CCI Landcover dataset."""

    _pretty = "Ext ESA Landcover"

    def __init__(self, start_year=1992, end_year=2020, max_workers=2):
        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        import cdsapi

        client = cdsapi.Client(quiet=True, progress=False, delete=False)

        def non_delete_retrieve_extract(year, *request):
            """Retrieve and extract a request without overwriting."""
            if os.path.exists(request[2]):
                logger.warning(
                    f"Target file '{request[2]}' already exists. Not downloading."
                )
            else:
                logger.info(f"Downloading to target file '{request[2]}'.")
                client.retrieve(*request)

            extract_dir = os.path.join(download_dir, f"extracted_{year}")

            if os.path.isdir(extract_dir):
                logger.warning(
                    f"Extract directory '{extract_dir}' already exists. "
                    "Not extracting."
                )
            else:
                logger.info(f"Extracting to directory '{extract_dir}'. ")
                # Unpack the downloaded archive.
                shutil.unpack_archive(
                    request[2],
                    extract_dir=extract_dir,
                )

        # Download data for processing.
        download_dir = os.path.join(tempfile.gettempdir(), self.name)
        if not os.path.isdir(download_dir):
            os.mkdir(download_dir)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        future_to_dest = {
            executor.submit(
                non_delete_retrieve_extract,
                year,
                "satellite-land-cover",
                {
                    "variable": "all",
                    "format": "tgz",
                    "version": [
                        "v2.0.7cds",
                        "v2.1.1",
                    ],
                    "year": [
                        f"{year}",
                    ],
                },
                os.path.join(download_dir, f"download_{year}.tar.gz"),
            ): (year, os.path.join(download_dir, f"extracted_{year}"))
            for year in range(start_year, end_year)
        }

        regridded_pft_cubes = iris.cube.CubeList()

        for future in tqdm(
            concurrent.futures.as_completed(future_to_dest),
            desc="Processing ESA CCI Landcover",
            total=len(future_to_dest),
        ):
            future.result()  # Ensure any raised Exceptions are raised again here.
            year, extract_dir = future_to_dest[future]
            # Cache the results of processing.
            processed_file = os.path.join(extract_dir, "processed.nc")
            if os.path.isfile(processed_file):
                logger.info(f"Loaded processed PFTs from '{processed_file}'.")
                regridded_pft_cubes.extend(iris.load(processed_file))
                # Do not carry out the normal processing below, as this is contained
                # in the loaded file.
                continue

            # Process the resulting file by converting the landcover categories to
            # PFTs, followed by regridding.
            logger.info(f"Loading landcover for year {year}.")
            category_cube = iris.load_cube(
                os.path.join(extract_dir, "*.nc"),
                constraint=iris.Constraint("land_cover_lccs"),
            )
            assert category_cube.shape[0] == 1, "There should only be one time."

            logger.info(f"Converting landcover to PFTs for year {year}.")

            # 18 MB for 2 GB worker memory.
            category_cube = category_cube[0]
            category_cube = category_cube.copy(
                data=category_cube.core_data().rechunk(("10MiB", -1))
            )

            print("core data:")
            print(category_cube.core_data())

            pft_cubes = convert_to_pfts(category_cube, lc_to_pft_map, 0, 220)

            year_regridded_pft_cubes = iris.cube.CubeList()

            logger.info(f"Regridding PFTs for year {year}.")
            # 110 MB for 15 GB worker memory (1 thread per worker).
            regrid_max_chunk_size = "110MB"
            for pft_cube in tqdm(pft_cubes, desc=f"Regridding ({year})"):
                tgt_cube = dummy_lat_lon_cube(da.zeros((720, 1440)))
                tgt_cube.metadata = pft_cube.metadata
                tgt_cube.add_aux_coord(pft_cube.coord("time"))

                year_regridded_pft_cubes.append(
                    spatial_chunked_regrid(
                        pft_cube,
                        tgt_cube,
                        iris.analysis.AreaWeighted(),
                        max_src_chunk_size=regrid_max_chunk_size,
                        max_tgt_chunk_size=regrid_max_chunk_size,
                    )
                )
            # Cache the data.
            logger.info("Realising processed PFTs data.")
            year_regridded_pft_cubes.realise_data()
            iris.save(year_regridded_pft_cubes, processed_file, zlib=False)
            logger.info(f"Cached processed PFTs in '{processed_file}'.")

            # Record the processed cubes.
            regridded_pft_cubes.extend(year_regridded_pft_cubes)

        executor.shutdown()

        # Process attributes that will differ from year to years to allow merging of
        # the cubes (id, tracking_id, creation_date, time_coverage_start,
        # time_coverage_end).
        merged_pft_cubes = iris.cube.CubeList()
        for pft_name in all_pfts:
            cubes = regridded_pft_cubes.extract(iris.Constraint(pft_name))
            assert len(cubes) == len(
                future_to_dest
            ), "There should be as many cubes as there are downloaded years."
            cube_ids = [cube.attributes["id"] for cube in cubes]
            cube_tracking_ids = [cube.attributes["tracking_id"] for cube in cubes]
            cube_creation_dates = [cube.attributes["creation_date"] for cube in cubes]
            cube_time_coverages = [
                "-".join(
                    (
                        cube.attributes["time_coverage_start"],
                        cube.attributes["time_coverage_end"],
                    )
                )
                for cube in cubes
            ]
            # Remove the original attributes.
            for attribute in (
                "id",
                "tracking_id",
                "creation_date",
                "time_coverage_start",
                "time_coverage_end",
            ):
                for cube in cubes:
                    del cube.attributes[attribute]
            # Replace the original attributes with aggregated versions.
            for cube in cubes:
                cube.attributes["ids"] = tuple(cube_ids)
                cube.attributes["tracking_ids"] = tuple(cube_tracking_ids)
                cube.attributes["creation_dates"] = tuple(cube_creation_dates)
                cube.attributes["time_coverages"] = tuple(cube_time_coverages)

            merged_pft_cubes.append(cubes.merge_cube())

        self.cubes = merged_pft_cubes
        assert len(self.cubes) == len(
            all_pfts
        ), "There should be as many cubes as PFTs."
        self.write_cache()

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.interpolate_yearly_data(start, end)


class ESA_CCI_Landcover_PFT(Dataset):
    _pretty = "ESA Landcover"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ESA-CCI-LC_landcover", "0d25_lc2pft")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        loaded_cubes = iris.load(os.path.join(self.dir, "*.nc"))

        time_coord = None
        for cube in loaded_cubes:
            if cube.coords()[0].name() == "time":
                time_coord = cube.coord("time")
                break
        assert time_coord.standard_name == "time"

        # fix peculiar 'z' coordinate, which should be the number of years
        for cube in loaded_cubes:
            coord_names = [coord.name() for coord in cube.coords()]
            if "z" in coord_names:
                assert coord_names[0] == "z"
                cube.remove_coord("z")
                cube.add_dim_coord(time_coord, 0)

        self.cubes = loaded_cubes

        self.time_unit_str = time_coord.units.name
        self.calendar = time_coord.units.calendar

        for cube in self.cubes:
            # Fill masked elements with the minimum value.
            fill_val = np.min(cube.data)
            cube.data.data[cube.data.mask] = fill_val
            logger.warning(f"Filling dataset: {self}, cube: {cube} with: {fill_val}")
            # Reset mask.
            cube.data.mask = np.zeros_like(cube.data.mask)

        self.write_cache()

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.interpolate_yearly_data(start, end)


class ESA_CCI_Soilmoisture(MonthlyDataset):
    _pretty = "ESA CCI Soil Moisture"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ESA-CCI-SM_soilmoisture")
        self.cubes = iris.load(os.path.join(self.dir, "*.nc"))


class ESA_CCI_Soilmoisture_Daily(Dataset):
    _pretty = "ESA CCI Daily Soil Moisture"
    _not_implemented = True

    def __init__(self):
        raise NotImplementedError("Use ESA_CCI_Soilmoisture Dataset for monthly data!")
        self.dir = os.path.join(DATA_DIR, "soil-moisture", "daily_files", "COMBINED")
        files = sorted(glob.glob(os.path.join(self.dir, "**", "*.nc")))
        raw_cubes = load_cubes(files, 100)

        # Delete varying attributes.
        for cube in raw_cubes:
            for attr in ["id", "tracking_id", "date_created"]:
                del cube.attributes[attr]

        # For the observation timestamp cubes, remove the 'valid_range'
        # attribute, which varies from cube to cube. The values of this
        # parameter are [-0.5, 0.5] for day 0, [0.5, 1.5] for day 1, etc...
        #
        # TODO: This seems to work but seems kind of hacky - is it really
        # guaranteed that the ordering of the cubes is constant?
        for cube in raw_cubes[7:None:8]:
            del cube.attributes["valid_range"]

        self.cubes = raw_cubes.concatenate()

        for cube in self.cubes:
            iris.coord_categorisation.add_month_number(cube, "time")
            iris.coord_categorisation.add_year(cube, "time")

        # Perform averaging over months in each year.
        self.monthly_means = iris.cube.CubeList()
        for cube in self.cubes:
            self.monthly_means.append(
                cube.aggregated_by(["month_number", "year"], iris.analysis.MEAN)
            )

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        raise NotImplementedError("Use ESA_CCI_Soilmoisture Dataset for monthly data!")
        # TODO: Isolate actual soil moisture.
        return self.monthly_means.extract(
            iris.Constraint(time=lambda t: end >= t.point >= start)
        )


class GFEDv4(MonthlyDataset):
    """Without small fires."""

    _pretty = "GFED4"
    pretty_variable_names = {"monthly burned area": "GFED4 BA"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "gfed4", "data")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        filenames = glob.glob(os.path.join(self.dir, "*MQ*.hdf"))
        filenames.sort()  # increasing months & years

        datetimes = []
        data = []

        for f in filenames:
            hdf = SD(f, SDC.READ)
            # TODO: Use 'BurnedAreaUncertainty' dataset, and maybe others,
            # like 'FirePersistence' (viewed using hdf.datasets()).
            burned_area = hdf.select("BurnedArea")

            attributes = burned_area.attributes()

            split_f = os.path.split(f)[1]
            year = int(split_f[11:15])
            month = int(split_f[15:17])

            assert 1990 < year < 2030
            assert 0 < month < 13

            datetimes.append(datetime(year, month, 1))
            data.append(
                burned_area[:][np.newaxis].astype("float64")
                * attributes["scale_factor"]
            )

        data = np.vstack(data)

        unit = cf_units.Unit(attributes["units"])
        long_name = attributes["long_name"]

        calendar = "gregorian"
        time_unit_str = "days since 1970-01-01 00:00:00"
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)
        time_coord = iris.coords.DimCoord(
            [cf_units.date2num(dt, time_unit_str, calendar) for dt in datetimes],
            standard_name="time",
            units=time_unit,
        )

        latitudes = iris.coords.DimCoord(
            get_centres(np.linspace(90, -90, 721)),
            standard_name="latitude",
            units="degrees",
        )
        longitudes = iris.coords.DimCoord(
            get_centres(np.linspace(-180, 180, 1441)),
            standard_name="longitude",
            units="degrees",
        )

        latitudes.guess_bounds()
        longitudes.guess_bounds()

        burned_area_cube = iris.cube.Cube(
            data,
            long_name=long_name,
            units=unit,
            dim_coords_and_dims=[(time_coord, 0), (latitudes, 1), (longitudes, 2)],
        )

        # Normalise using the grid cell areas, divide by 10000 to convert the scaling
        # factors from m2 to hectares (the burned areas are in hectares originally).
        # NOTE: Some burned area fractions may be above 1!
        burned_area_cube.data /= (
            iris.analysis.cartography.area_weights(burned_area_cube) / 10000
        )
        burned_area_cube.units = cf_units.Unit(1)

        self.cubes = iris.cube.CubeList([burned_area_cube])
        self.write_cache()


class GFEDv4s(MonthlyDataset):
    """Includes small fires."""

    _pretty = "GFED4s"
    pretty_variable_names = {"Burnt_Area": "GFED4s BA"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "gfed4", "data")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        filenames = glob.glob(os.path.join(self.dir, "*.hdf5"))
        filenames.sort()  # increasing years

        # for each file (each year), load the data, the latitudes and
        # longitudes and place them into a cube
        years = []
        data = []
        for f in filenames:
            year = int(f[-9:-5])
            years.append(year)
            container = h5py.File(f, mode="r")

            for month_str in [format(m, "02d") for m in range(1, 13)]:
                data.append(
                    container["burned_area"][month_str]["burned_fraction"][()][
                        None, ...
                    ]
                )

        assert years == sorted(years), "Should be monotonically increasing"

        # use the last file (of previous for loop) to get latitudes and
        # longitudes, assuming that they are the same for all the data
        # files!
        latitudes = container["lat"][()]
        longitudes = container["lon"][()]

        # make sure that the lats and lons are uniform along the grid
        assert np.all(longitudes[0] == longitudes)
        assert np.all(latitudes.T[0] == latitudes.T)

        longitudes = iris.coords.DimCoord(
            longitudes[0], standard_name="longitude", units="degrees"
        )
        latitudes = iris.coords.DimCoord(
            latitudes.T[0], standard_name="latitude", units="degrees"
        )

        time_unit_str = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        num_times = []
        for m in range(len(data)):
            month = (m % 12) + 1
            year = (m // 12) + min(years)
            assert year <= max(years)
            num_times.append(
                cf_units.date2num(datetime(year, month, 1), time_unit_str, calendar)
            )

        time_coord = iris.coords.DimCoord(
            num_times, standard_name="time", units=time_unit
        )

        for coord in (longitudes, latitudes, time_coord):
            coord.guess_bounds()

        self.cubes = iris.cube.CubeList(
            [
                iris.cube.Cube(
                    np.vstack(data),
                    dim_coords_and_dims=[
                        (time_coord, 0),
                        (latitudes, 1),
                        (longitudes, 2),
                    ],
                )
            ]
        )

        self.cubes[0].units = cf_units.Unit(1)
        self.cubes[0].var_name = "Burnt_Area"
        self.write_cache()


class GlobFluo_SIF(MonthlyDataset):
    _pretty = "Glob Fluo SIF"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "GlobFluo_SIF")
        loaded_cube = iris.load_cube(os.path.join(self.dir, "*.nc"))

        # Need to convert to time coordinate, as values are relative to
        # 1582-10-14, which is not supported by the cf_units gregorian
        # calendar (needs to start from 1582-10-15, I think).

        # Get the original number of days relative to 1582-10-14 00:00:00.
        days_since_1582_10_14 = loaded_cube.coords()[0].points
        # Define new time unit relative to a supported date.
        new_time_unit = cf_units.Unit(
            "days since 1582-10-16 00:00:00", calendar="gregorian"
        )
        # The corresponding number of days for the new time unit.
        days_since_1582_10_16 = days_since_1582_10_14 - 2

        loaded_cube.remove_coord("time")
        new_time = iris.coords.DimCoord(
            days_since_1582_10_16, standard_name="time", units=new_time_unit
        )
        loaded_cube.add_dim_coord(new_time, 0)

        invalid_mask = np.logical_or(
            loaded_cube.data.data > 20, loaded_cube.data.data < 0
        )
        logger.info(f"Masking {np.sum(invalid_mask)} invalid values for SIF.")
        loaded_cube.data.mask |= invalid_mask

        self.cubes = iris.cube.CubeList([loaded_cube])


class GPW_v4_pop_dens(Dataset):
    _pretty = "GPW4 Pop Density"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "GPW_v4_pop_dens")
        netcdf_dataset = netCDF4.Dataset(glob.glob(os.path.join(self.dir, "*.nc"))[0])
        data = netcdf_dataset[
            "Population Density, v4.10 (2000, 2005, 2010,"
            " 2015, 2020): 30 arc-minutes"
        ]

        datetimes = [datetime(year, 1, 1) for year in [2000, 2005, 2010, 2015, 2020]]
        self.time_unit_str = "days since {:}".format(str(datetime(1970, 1, 1)))
        self.calendar = "gregorian"
        self.time_unit = cf_units.Unit(self.time_unit_str, calendar=self.calendar)
        time = iris.coords.DimCoord(
            cf_units.date2num(datetimes, self.time_unit_str, calendar="gregorian"),
            standard_name="time",
            units=self.time_unit,
        )

        latitudes = iris.coords.DimCoord(
            netcdf_dataset["latitude"][:], standard_name="latitude", units="degrees"
        )
        longitudes = iris.coords.DimCoord(
            netcdf_dataset["longitude"][:], standard_name="longitude", units="degrees"
        )

        coords = [(time, 0), (latitudes, 1), (longitudes, 2)]

        self.cubes = iris.cube.CubeList(
            [
                iris.cube.Cube(
                    data[:5],
                    long_name=data.long_name,
                    var_name="Population_Density",
                    units=cf_units.Unit("1/km2"),
                    dim_coords_and_dims=coords,
                )
            ]
        )

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        """Linear interpolation onto the target months."""
        final_cubelist = self.interpolate_yearly_data(start, end)
        assert len(final_cubelist) == 1
        return final_cubelist


class GSMaP_dry_day_period(MonthlyDataset):
    """Calculate the length of the longest preceding dry day period.

    This definition only considers dry day periods within the current month, or dry
    day periods that occur within the current month AND previous months, ONLY if these
    join up contiguously at the month boundaries.

    Other definitions taking into account (only) dry day periods in a certain number
    of months leading up to the current month may be possible as well, although this
    could also be implemented in post-processing.

    """

    _pretty = "GSMaP Dry Day Period"
    pretty_variable_names = {"dry_day_period": "Dry Day Period"}

    def __init__(self, times="00Z-23Z"):
        self.dir = os.path.join(
            DATA_DIR,
            "GSMaP_Precipitation",
            "hokusai.eorc.jaxa.jp",
            "realtime_ver",
            "v6",
            "daily_G",
            times,
        )

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        # Sort so that time is increasing.
        filenames = sorted(glob.glob(os.path.join(self.dir, "**", "*.nc")))

        logger.info("Constructing dry day period cube.")
        dry_day_period_cubes = iris.cube.CubeList()

        prev_dry_day_period = None
        prev_end = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Collapsing a non-contiguous coordinate. Metadata may not "
                    "be fully descriptive for 'time'."
                ),
            )
            for filename in tqdm(filenames):
                # Clip outer values which are duplicated in the data
                # selection below and not needed here.
                raw_cube = iris.load_cube(filename)[..., 1:1441]
                n_days = raw_cube.shape[0]
                n_lats = raw_cube.shape[1]
                n_lons = raw_cube.shape[2]

                # The first time around only, create empty arrays. This will introduce
                # some negative bias for the first month(s), but this should be
                # negligible overall (especially since the first year is probably not
                # being used anyway).
                if prev_dry_day_period is None:
                    assert prev_end is None
                    prev_dry_day_period = np.zeros((n_lats, n_lons), dtype=np.int64)
                    prev_end = np.zeros((n_lats, n_lons), dtype=np.bool_)

                longitude_points = raw_cube.coord("longitude").points
                assert np.min(longitude_points) == 0.125
                assert np.max(longitude_points) == 359.875

                # No need to calculate mean cube here, as we are only interested in
                # the raw daily precipitation data.

                # Calculate dry days.
                dry_days = raw_cube.data < MM_PER_HR_THRES

                # Find contiguous blocks in the time dimension where dry_days is True.
                structure = np.zeros((3, 3, 3), dtype=np.int64)
                structure[:, 1, 1] = 1
                labelled = scipy.ndimage.label(dry_days, structure=structure)
                slices = scipy.ndimage.find_objects(labelled[0])

                dry_day_period = np.zeros((n_lats, n_lons), dtype=np.int64)
                beginning = np.zeros((n_lats, n_lons), dtype=np.bool_)
                end = np.zeros_like(beginning)

                for slice_object in slices:
                    time_slice = slice_object[0]
                    lat_slice = slice_object[1]
                    lon_slice = slice_object[2]
                    assert lat_slice.stop - lat_slice.start == 1
                    assert lon_slice.stop - lon_slice.start == 1

                    latitude = lat_slice.start
                    longitude = lon_slice.start

                    period_length = time_slice.stop - time_slice.start

                    if period_length > dry_day_period[latitude, longitude]:
                        dry_day_period[latitude, longitude] = period_length
                        if time_slice.start == 0:
                            beginning[latitude, longitude] = True
                        else:
                            beginning[latitude, longitude] = False
                        if time_slice.stop == n_days:
                            end[latitude, longitude] = True
                        else:
                            end[latitude, longitude] = False

                # Once the data for the current month has been processed, look at the
                # previous month to see if dry day periods may be joined up.
                overlap = prev_end & beginning
                dry_day_period[overlap] += prev_dry_day_period[overlap]

                # Prepare for the next month's analysis.
                prev_dry_day_period = dry_day_period
                prev_end = end

                # Create new Cube with the same latitudes and longitudes, and an
                # averaged time.
                coords = [
                    (raw_cube.coord("latitude"), 0),
                    (raw_cube.coord("longitude"), 1),
                ]

                # Modify the time coordinate such that it is recorded with
                # respect to a common date, as opposed to relative to the
                # beginning of the respective month as is the case for the
                # cube loaded above.

                # Take the new 'mean' time as the average of the first and last time.
                min_time = raw_cube.coord("time").cell(0).point
                max_time = raw_cube.coord("time").cell(-1).point
                centre_datetime = min_time + ((max_time - min_time) / 2)

                new_time = cf_units.date2num(
                    centre_datetime, self.time_unit_str, self.calendar
                )
                time_coord = iris.coords.DimCoord(
                    new_time, units=self.time_unit, standard_name="time"
                )

                dry_day_period_cube = iris.cube.Cube(
                    dry_day_period,
                    dim_coords_and_dims=coords,
                    units=cf_units.Unit("days"),
                    var_name="dry_day_period",
                    aux_coords_and_dims=[(time_coord, None)],
                )
                dry_day_period_cube.units = cf_units.Unit("days")

                dry_day_period_cubes.append(dry_day_period_cube)

        self.cubes = iris.cube.CubeList([dry_day_period_cubes.merge_cube()])
        self.write_cache()


class GSMaP_precipitation(MonthlyDataset):
    _pretty = "GSMaP Precipitation"
    pretty_variable_names = {"dry_days": "Dry Days", "precip": "Precipitation"}

    def __init__(self, times="00Z-23Z"):
        self.dir = os.path.join(
            DATA_DIR,
            "GSMaP_Precipitation",
            "hokusai.eorc.jaxa.jp",
            "realtime_ver",
            "v6",
            "daily_G",
            times,
        )

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        # Sort so that time is increasing.
        filenames = sorted(glob.glob(os.path.join(self.dir, "**", "*.nc")))

        calendar = "gregorian"
        time_unit_str = "days since 1970-01-01 00:00:00"
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        logger.info("Constructing average precipitation and dry days cubes.")
        monthly_average_cubes = iris.cube.CubeList()
        dry_days_cubes = iris.cube.CubeList()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Collapsing a non-contiguous coordinate. Metadata may not "
                    "be fully descriptive for 'time'."
                ),
            )
            for filename in tqdm(filenames):
                # Clip outer values which are duplicated in the data
                # selection below and not needed here.
                raw_cube = iris.load_cube(filename)[..., 1:1441]
                monthly_cube = raw_cube.collapsed("time", iris.analysis.MEAN)

                longitude_points = monthly_cube.coord("longitude").points
                assert np.min(longitude_points) == 0.125
                assert np.max(longitude_points) == 359.875

                # Modify the time coordinate such that it is recorded with
                # respect to a common date, as opposed to relative to the
                # beginning of the respective month as is the case for the
                # cube loaded above.
                centre_datetime = monthly_cube.coord("time").cell(0).point
                new_time = cf_units.date2num(centre_datetime, time_unit_str, calendar)
                monthly_cube.coord("time").bounds = None
                monthly_cube.coord("time").points = [new_time]
                monthly_cube.coord("time").units = time_unit

                monthly_cube.units = cf_units.Unit("mm/hr")

                monthly_average_cubes.append(monthly_cube)

                # Calculate dry day statistics.

                dry_days_data = np.sum(raw_cube.data < MM_PER_HR_THRES, axis=0)

                coords = [
                    (monthly_cube.coord("latitude"), 0),
                    (monthly_cube.coord("longitude"), 1),
                ]
                dry_days_cubes.append(
                    iris.cube.Cube(
                        dry_days_data,
                        dim_coords_and_dims=coords,
                        units=cf_units.Unit("days"),
                        var_name="dry_days",
                        aux_coords_and_dims=[(monthly_cube.coord("time"), None)],
                    )
                )

        self.cubes = iris.cube.CubeList(
            [monthly_average_cubes.merge_cube(), dry_days_cubes.merge_cube()]
        )
        assert len(self.cubes) == 2

        self.write_cache()


class HYDE(Dataset):
    _pretty = "HYDE"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "HYDE")

        self.time_unit_str = "hours since 1970-01-01 00:00:00"
        self.calendar = "gregorian"
        self.time_unit = cf_units.Unit(self.time_unit_str, calendar=self.calendar)

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        # TODO: Consider upper and lower estimates as well, not just
        # baseline??
        files = glob.glob(os.path.join(self.dir, "baseline", "*.asc"), recursive=True)

        cube_list = iris.cube.CubeList()
        mapping = {
            "uopp": {},
            "urbc": {},
            "tot_rice": {},
            "tot_rainfed": {},
            "tot_irri": {},
            "rurc": {},
            "rf_rice": {},
            "rf_norice": {},
            "rangeland": {},
            "popd": {},
            "popc": {},
            "pasture": {},
            "ir_rice": {},
            "ir_norice": {},
            "grazing": {},
            "cropland": {},
            "conv_rangeland": {},
        }
        pattern = re.compile(r"(.*)(\d{4})AD")

        for f in tqdm(files):
            groups = pattern.search(os.path.split(f)[1]).groups()
            variable_key = groups[0].strip("_")
            year = int(groups[1])
            data = np.loadtxt(f, skiprows=6, ndmin=2)
            assert data.shape == (2160, 4320)
            data = data.reshape(2160, 4320)
            data = np.ma.MaskedArray(data, mask=np.isclose(data, -9999))

            new_latitudes = get_centres(np.linspace(90, -90, data.shape[0] + 1))
            new_longitudes = get_centres(np.linspace(-180, 180, data.shape[1] + 1))
            new_lat_coord = iris.coords.DimCoord(
                new_latitudes, standard_name="latitude", units="degrees"
            )
            new_lon_coord = iris.coords.DimCoord(
                new_longitudes, standard_name="longitude", units="degrees"
            )

            grid_coords = [(new_lat_coord, 0), (new_lon_coord, 1)]

            time_coord = iris.coords.DimCoord(
                cf_units.date2num(
                    datetime(year, 1, 1), self.time_unit_str, self.calendar
                ),
                standard_name="time",
                units=self.time_unit,
            )

            cube = iris.cube.Cube(
                data,
                dim_coords_and_dims=grid_coords,
                units=mapping[variable_key].get("unit"),
                var_name=variable_key,
                long_name=mapping[variable_key].get("long_name"),
                aux_coords_and_dims=[(time_coord, None)],
            )
            regrid_cube = regrid(cube)
            cube_list.append(regrid_cube)

        self.cubes = cube_list.merge()
        self.write_cache()

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        """Linear interpolation onto the target months."""
        return self.interpolate_yearly_data(start, end)


class LIS_OTD_lightning_climatology(Dataset):
    _pretty = "LIS/OTD"
    pretty_variable_names = {
        "Combined Flash Rate Monthly Climatology": "Lightning Climatology"
    }

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "LIS_OTD_lightning_climatology")

        loaded_cube = iris.load(os.path.join(self.dir, "*.nc")).extract_cube(
            iris.Constraint(name="Combined Flash Rate Monthly Climatology")
        )

        # Fix time units so they do not refer months, as this can't be processed by
        # iris / cf_units.
        # Realign times so they are at the centre of each month.

        # Check that existing time coordinate is as expected.
        assert loaded_cube.coord("time").units.origin == "months since 2014-1-1 0:0:0"
        assert all(loaded_cube.coord("time").points == np.arange(1, 13))

        datetimes = [
            (
                (
                    (datetime(2014, month, 1) + relativedelta(months=+1))
                    - datetime(2014, month, 1)
                )
                / 2
            )
            + datetime(2014, month, 1)
            for month in np.arange(1, 13)
        ]
        time_unit_str = "days since {:}".format(str(datetime(2014, 1, 1)))
        time_unit = cf_units.Unit(time_unit_str, calendar="gregorian")
        time_coord = iris.coords.DimCoord(
            cf_units.date2num(datetimes, time_unit_str, calendar="gregorian"),
            standard_name="time",
            units=time_unit,
        )
        loaded_cube.coord("time").points = time_coord.points
        loaded_cube.coord("time").units = time_coord.units

        invalid_mask = loaded_cube.data.data < 0
        logger.info(f"Masking {np.sum(invalid_mask)} invalid values for LIS/OTD.")
        loaded_cube.data.mask |= invalid_mask

        self.cubes = iris.cube.CubeList([loaded_cube])

        # Make sure that the time coordinate is the first coordinate.
        self.cubes = self.get_monthly_data(
            start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
        )

    @property
    def frequency(self):
        return "monthly climatology"

    @property
    def min_time(self):
        # FIXME: Find beginning of data validity!
        return "N/A"

    @property
    def max_time(self):
        # FIXME: Find end of data validity!
        return "N/A"

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        """'Broadcast' monthly climatology across the requested time
        period.

        NOTE: This is a climatology, and not true monthly data!

        This method ignores days.

        """
        # TODO: Make this work with lazy data?

        cube = self.cube
        assert (
            len(cube.coord("time").points) == 12
        ), "Only meant to be run starting from the initial state, which as 12 months."

        # Time index will vary from the first run (simply re-shuffling the coordinate
        # order) to the second run (which will then actually expand the months to the
        # desired range).
        time_index = cube.coords().index(cube.coord("time"))

        datetimes = [datetime(start.year, start.month, 1)]
        while datetimes[-1] != PartialDateTime(end.year, end.month):
            datetimes.append(datetimes[-1] + relativedelta(months=+1))

        output_arrs = []

        for dt in datetimes:
            selection = [slice(None)] * 3
            selection[time_index] = (dt.month - 1) % 12
            output_arrs.append(cube[tuple(selection)].data[np.newaxis])

        output_data = np.vstack(output_arrs)

        time_coord = iris.coords.DimCoord(
            cf_units.date2num(datetimes, self.time_unit_str, calendar=self.calendar),
            standard_name="time",
            units=self.time_unit,
        )

        new_coords = [
            (time_coord, 0),
            (cube.coord("latitude"), 1),
            (cube.coord("longitude"), 2),
        ]

        output_cube = iris.cube.Cube(
            output_data,
            dim_coords_and_dims=new_coords,
            standard_name=cube.standard_name,
            long_name=cube.long_name,
            var_name=cube.var_name,
            units=cube.units,
            attributes=cube.attributes,
        )

        return iris.cube.CubeList([output_cube])


class LIS_OTD_lightning_time_series(MonthlyDataset):
    _pretty = "LIS/OTD Time Series"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "LIS_OTD_lightning_time_series")

        self.cubes = self.read_cache()
        # Exit __init__ if we have loaded the data.
        if self.cubes:
            return

        # Otherwise keep loading the data.
        raw_cubes = iris.load(os.path.join(self.dir, "*.nc"))
        # TODO: Use other attributes as well? Eg. separate LIS / OTD data,
        # grid cell area, or Time Series Sampling (km^2 / day)?

        # Isolate single combined flash rate.
        raw_cubes = raw_cubes.extract(
            iris.Constraint(name="Combined Flash Rate Time Series")
        )

        for cube in raw_cubes:
            iris.coord_categorisation.add_month_number(cube, "time")
            iris.coord_categorisation.add_year(cube, "time")

        monthly_cubes = [
            cube.aggregated_by(["month_number", "year"], iris.analysis.MEAN)
            for cube in raw_cubes
        ]

        # Create new cube(s) where the time dimension is the first
        # dimension. To do this, the cube metadata can be copied, while new
        # coordinates and corresponding data (both simply
        # reshaped/reordered) are assigned.

        new_coords = [
            (monthly_cubes[0].coord("time"), 0),
            (monthly_cubes[0].coord("latitude"), 1),
            (monthly_cubes[0].coord("longitude"), 2),
        ]

        self.cubes = iris.cube.CubeList()
        for cube in monthly_cubes:
            # NOTE: This does not use any lazy data whatsoever, starting
            # with the monthly aggregation above.
            assert cube.shape[-1] == len(
                cube.coord("time").points
            ), "Old and new time dimension should have the same length"
            data_arrs = []
            for time_index in range(cube.shape[-1]):
                data_arrs.append(cube.data[..., time_index][np.newaxis])

            new_data = np.ma.vstack(data_arrs)

            new_cube = iris.cube.Cube(new_data, dim_coords_and_dims=new_coords)
            new_cube.metadata = deepcopy(cube.metadata)
            self.cubes.append(new_cube)

        self.write_cache()


class Liu_VOD(MonthlyDataset):
    _pretty = "Liu VOD"
    pretty_variable_names = {"VODorig": "VOD"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "Liu_VOD")
        loaded_cubes = iris.cube.CubeList(
            [iris.load_cube(os.path.join(self.dir, "*.nc"))]
        )

        # Need to convert to time coordinate, as values are relative to
        # 1582-10-14, which is not supported by the cf_units gregorian
        # calendar (needs to start from 1582-10-15, I think).

        # Get the original number of days relative to 1582-10-14 00:00:00.
        days_since_1582_10_14 = loaded_cubes[0].coords()[0].points
        # Define new time unit relative to a supported date.
        new_time_unit = cf_units.Unit(
            "days since 1582-10-16 00:00:00", calendar="gregorian"
        )
        # The corresponding number of days for the new time unit.
        days_since_1582_10_16 = days_since_1582_10_14 - 2

        loaded_cubes[0].remove_coord("time")
        new_time = iris.coords.DimCoord(
            days_since_1582_10_16, standard_name="time", units=new_time_unit
        )
        loaded_cubes[0].add_dim_coord(new_time, 0)
        self.cubes = loaded_cubes


class MCD64CMQ_C6(MonthlyDataset):
    _pretty = "MCD64CMQ C6"
    pretty_variable_names = {"Burned Area": "MCD64CMQ BA"}

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "MCD64CMQ_C6")

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        filenames = glob.glob(os.path.join(self.dir, "*MQ*.hdf"))
        filenames.sort()  # increasing months & years

        datetimes = []
        data = []

        for f in filenames:
            hdf = SD(f, SDC.READ)
            # TODO: Use 'QA' and 'UnmappedFraction' datasets (see hdf.datasets()).

            burned_area = hdf.select("BurnedArea")

            split_f = os.path.split(f)[1].split(".")[1][1:]
            year = int(split_f[:4])
            day = int(split_f[4:])

            date = datetime(year, 1, 1) + timedelta(day - 1)

            assert 2000 <= date.year <= 2030
            assert 0 < date.month < 13

            datetimes.append(date)
            data.append(
                burned_area[:][np.newaxis].astype("float64")
                # Scale factor from MODIS_C6_BA_User_Guide_1.2, August 2018, to
                # yield burnt area in hectares.
                * 0.01
            )

        data = np.vstack(data)

        time_coord = iris.coords.DimCoord(
            [
                cf_units.date2num(dt, self.time_unit_str, self.calendar)
                for dt in datetimes
            ],
            standard_name="time",
            units=self.time_unit,
        )

        latitudes = iris.coords.DimCoord(
            get_centres(np.linspace(90, -90, 721)),
            standard_name="latitude",
            units="degrees",
        )
        longitudes = iris.coords.DimCoord(
            get_centres(np.linspace(-180, 180, 1441)),
            standard_name="longitude",
            units="degrees",
        )

        latitudes.guess_bounds()
        longitudes.guess_bounds()

        burned_area_cube = iris.cube.Cube(
            data,
            long_name="Burned Area",
            dim_coords_and_dims=[(time_coord, 0), (latitudes, 1), (longitudes, 2)],
        )

        # Normalise using the grid cell areas, divide by 10000 to convert the scaling
        # factors from m2 to hectares (the burned areas are in hectares originally).
        # NOTE: Some burned area fractions may be above 1!
        burned_area_cube.data /= (
            iris.analysis.cartography.area_weights(burned_area_cube) / 10000
        )

        burned_area_cube.units = cf_units.Unit(1)

        self.cubes = iris.cube.CubeList([burned_area_cube])
        self.write_cache()


class MOD15A2H_LAI_fPAR(MonthlyDataset):
    _pretty = "MOD15A2H"
    pretty_variable_names = {
        "Fraction of Absorbed Photosynthetically Active Radiation": "FAPAR",
        "Leaf Area Index": "LAI",
    }

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "MOD15A2H_LAI-fPAR")
        self.cubes = iris.load(os.path.join(self.dir, "*.nc"))

        months = []
        for i in range(self.cubes[0].shape[0]):
            months.append(self.cubes[0].coords()[0].cell(i).point.month)

        assert np.all(
            np.diff(np.where(np.diff(months) != 1)) == 12
        ), "The year should increase every 12 samples!"

        # TODO: Since the day in the month for which the data is provided
        # is variable, take into account neighbouring months as well in a
        # weighted average (depending on how many days away from the middle
        # of the month these other samples are)?


class Simard_canopyheight(Dataset):
    _pretty = "Simard Canopy Height"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "Simard_canopyheight")
        self.cubes = iris.cube.CubeList(
            [iris.load_cube(os.path.join(self.dir, "*.nc"))]
        )

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.broadcast_static_data(start, end)


class Thurner_AGB(Dataset):
    _pretty = "Thurner AGB"
    # TODO: Look at data values - seems like there is a major issue there!

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "Thurner_AGB")
        # Ignore warning about units, which are fixed below.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(r"Ignoring netCDF variable.*invalid units 'kg\[C]/m2'"),
            )

            self.cubes = iris.load(os.path.join(self.dir, "*.nc"))

        for cube in self.cubes:
            cube.units = cf_units.Unit("kg(C)/m2")

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.broadcast_static_data(start, end)


class VODCA(MonthlyDataset):
    """Global VOD Dataset.

    See: https://zenodo.org/record/2575599#.XO6qXHVKibI

    """

    _pretty = "VODCA"
    pretty_variable_names = {
        "Vegetation optical depth Ku-band (18.7 GHz - 19.35 GHz)": "VOD Ku-band",
        "Vegetation optical depth X-band (10.65 GHz - 10.7 GHz)": "VOD X-band",
    }

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "VODCA")

        self.cubes = self.read_cache()
        # Exit __init__ if we have loaded the data.
        if self.cubes:
            return

        daily_dirs = glob.glob(os.path.join(self.dir, "daily", "*", "*"))
        # Calculate monthly averages using the daily data.
        assert all(len(os.path.split(dir_name)[1]) == 4 for dir_name in daily_dirs)

        mean_cubes = iris.cube.CubeList(
            # TODO: Check if using multi-processing here instead of using multiple
            # threads has the potential to speed up the averaging.
            # Parallel(n_jobs=get_ncpus(), prefer="threads")(
            Parallel(n_jobs=1, prefer="threads")(
                delayed(self._monthly_average_in_dir)(directory)
                for directory in tqdm(daily_dirs)
            )
        )

        mean_cubes = mean_cubes.concatenate()

        for cube in mean_cubes:
            # Add the band name to the cube name to prevent all variables (cubes)
            # having the same name, ie. to differentiate the cubes.
            cube.long_name = f"{cube.long_name} {cube.attributes['band']}"

        # TODO: Isolate different VOD bands, ignore masks (maybe put in different
        # `Dataset` instance?)

        self.cubes = mean_cubes

        self.write_cache()

    @staticmethod
    def _monthly_average_in_dir(directory):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=((r".*'vod' invalid units 'unitless'.*"))
            )
            warnings.filterwarnings(
                "ignore", message=((r".*'calendar' is not a permitted attribute.*"))
            )
            raw_cubes = iris.load(os.path.join(directory, "*.nc"))
            raw_cubes = iris.cube.CubeList(
                [
                    cube
                    for cube in raw_cubes
                    if "vegetation optical depth" in cube.name().lower()
                ]
            )
            raw_cubes = raw_cubes.concatenate()
            assert len(raw_cubes) == 1
            raw_cube = raw_cubes[0]
            iris.coord_categorisation.add_month_number(raw_cube, "time")
            iris.coord_categorisation.add_year(raw_cube, "time")

        return raw_cube.aggregated_by(["month_number", "year"], iris.analysis.MEAN)


class WWLLN(MonthlyDataset):
    _pretty = "WWLLN Lightning"
    pretty_variable_names = {
        "frequency of lightning flashes per unit area": "lightning"
    }

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "WWLLN")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=((r".*'lightning' invalid units 'strokes km-2'.*"))
            )
            raw_cube = iris.load_cube(os.path.join(self.dir, "WWLLN_monthly.nc"))
        raw_cube.units = cf_units.Unit("1/km2")
        self.cubes = iris.cube.CubeList([raw_cube])


def get_implemented_datasets(
    pretty_dataset_names=None, pretty_variable_names=None, ignore_names=IGNORED_DATASETS
):
    """Get all valid datasets defined in the `wildfires.data.datasets` module.

    Args:
        pretty_variable_names (dict): Dictionary mapping raw to pretty variable names
            ({raw: pretty, ...}).

    Returns:
        list of `Dataset`: List of datasets.

    """
    # TODO: Implement pretty dataset and variable names.

    if pretty_dataset_names is None:
        pretty_dataset_names = {}
    if pretty_variable_names is None:
        pretty_variable_names = {}
    if ignore_names is None:
        ignore_names = []
    dataset_list = []
    for dataset_cls in Dataset:
        if str(dataset_cls) in ignore_names:
            logger.debug(f"Ignoring {dataset_cls}.")
            continue
        try:
            dataset_list.append(dataset_cls())
        except NotImplementedError:
            logger.info(f"{dataset_cls} is not implemented.")
    return dataset_list


def dataset_times(
    datasets=None, dataset_names=None, lat_lon=False, bounds_from_monthly=True
):
    """Compile dataset time span information.

    Args:
        datasets (iterable of `Dataset`): If no value is given, defaults to using all
            available datasets.
        dataset_names (iterable of `str` or None): The names used for the datasets,
            the number of which should match the number of items in `datasets`. If
            None, use the `dataset.pretty` name for each `Dataset` in `datasets`.
        lat_lon (bool): Collect lat/lon grid information.
        bounds_from_monthly (bool): Only use monthly datasets to set the minimum and
            maximum temporal bounds.

    Returns:
        If valid starting and end times can be found for at least one of the datasets:
            - min_time: Minimum shared time of all datasets.
            - max_time: Maximum shared time of all datasets.
            - times_df: Pandas DataFrame encapsulating the timespan information.
        Otherwise the 3-tuple (None, None, None) will be returned.

    """
    if datasets is None:
        datasets = get_implemented_datasets()
    if dataset_names is None:
        dataset_names = tuple(dataset.pretty for dataset in datasets)

    attributes = ["min_time", "max_time", "frequency"]
    if lat_lon:
        attributes.extend(("lat_grid", "lon_grid"))

    time_dict = OrderedDict(
        (name, list(map(str, (getattr(dataset, attr) for attr in attributes))))
        for dataset, name in zip(datasets, dataset_names)
    )
    min_times, max_times = [], []
    for dataset in datasets:
        if dataset.frequency in ("static", "yearly", "climatology"):
            continue
        dataset_times = tuple(
            getattr(dataset, time_type) for time_type in ("min_time", "max_time")
        )
        # If there are any undefined dates they will be represented by strings and
        # should be skipped here.
        if any(isinstance(dataset_time, str) for dataset_time in dataset_times):
            assert all(
                isinstance(dataset_time, str) for dataset_time in dataset_times
            ), (
                "If there is no valid start date, there should not be a valid "
                "end date and vice versa (Dataset={}).".format(dataset)
            )
            continue
        assert (
            dataset_times[0] < dataset_times[1]
        ), "Maximum date should be after the minimum date (Dataset={}).".format(dataset)

        min_times.append(dataset_times[0])
        max_times.append(dataset_times[1])

    if not min_times and not max_times:
        logger.debug("No valid start or end times found.")
        return None, None, None

    # This timespan will encompass all the datasets.
    min_time = np.max(min_times)
    max_time = np.min(max_times)

    overall_placeholders = ["N/A"]
    if lat_lon:
        overall_placeholders.extend(("N/A", "N/A"))

    time_dict["Overall"] = list(map(str, (min_time, max_time, *overall_placeholders)))

    dataset_names = pd.Series(list(time_dict.keys()), name="Dataset")

    df_names = ["Minimum", "Maximum", "Frequency"]
    if lat_lon:
        df_names.extend(("Latitude Grid", "Longitude Grid"))

    df_series = [dataset_names]
    df_series.extend(
        [
            pd.Series([time_dict[name][i] for name in dataset_names], name=df_name)
            for i, df_name in enumerate(df_names)
        ]
    )

    times_df = pd.DataFrame(df_series).T

    if min_time >= max_time:
        limited_df = times_df[:-1]
        min_mask = limited_df["Minimum"].values.astype("str") == str(min_time)
        max_mask = limited_df["Maximum"].values.astype("str") == str(max_time)
        raise ValueError(
            "Maximum date should be after the minimum date. This suggests the datasets "
            "are improperly selected. Offending datasets:\n{}".format(
                limited_df.loc[min_mask | max_mask].to_string(index=False)
            )
        )

    return min_time, max_time, times_df


def regions_GFED():
    """Return cube describing the geographic regions used in GFED."""
    regions = dummy_lat_lon_cube(
        h5py.File(
            os.path.join(DATA_DIR, "gfed4", "data", "GFED4.1s_1997.hdf5"), mode="r"
        )["ancill"]["basis_regions"][:][::-1]
    )
    regions.long_name = "Basis-regions used for GFED analyses"
    regions.attributes["regions"] = {
        0: "Ocean",
        1: "BONA (Boreal North America)",
        2: "TENA (Temperate North America)",
        3: "CEAM (Central America)",
        4: "NHSA (Northern Hemisphere South America)",
        5: "SHSA (Southern Hemisphere South America)",
        6: "EURO (Europe)",
        7: "MIDE (Middle East)",
        8: "NHAF (Northern Hemisphere Africa)",
        9: "SHAF (Southern Hemisphere Africa)",
        10: "BOAS (Boreal Asia)",
        11: "CEAS (Central Asia)",
        12: "SEAS (Southeast Asia)",
        13: "EQAS (Equatorial Asia)",
        14: "AUST (Australia and New Zealand)",
    }
    regions.attributes["short_regions"] = {
        0: "Ocean",
        1: "BONA",
        2: "TENA",
        3: "CEAM",
        4: "NHSA",
        5: "SHSA",
        6: "EURO",
        7: "MIDE",
        8: "NHAF",
        9: "SHAF",
        10: "BOAS",
        11: "CEAS",
        12: "SEAS",
        13: "EQAS",
        14: "AUST",
    }
    # Invert the previous mapping to facilitate masking later.
    regions.attributes["region_codes"] = {
        code: index for index, code in regions.attributes["short_regions"].items()
    }
    return regions


# Automatically export all Dataset subclass leaves defining individual datasets.
__all__ = list(set(__all__).union(set(map(str, Dataset))))
