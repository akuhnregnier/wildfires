#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module that simplifies use of various datasets.

TODO:
    Figure out how to handle the fact that saving a cube for certain kinds of
    longitude coordinates adds the attribute `circular=True`.

    Since longitudes registered in the [-180, 180] system fail to register as circular
    using `iris.util._is_circular` but the same coordiantes + 180° do register
    correctly (if they actually cover all of [0, 360], as expected), does regridding
    using longitudes in [-180, 180] actually consider the coordinate's circular nature
    correctly?
    See also NOTE in `lat_lon_dimcoords`.

    Before caching, regrid() always needs to be called since this also carries out
    crucial coordinate attribute regularisation, which is essential for consistent
    caching behaviour! Make this intrinsic to __init__?

"""
import glob
import logging
import logging.config
import operator
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy
from datetime import datetime
from functools import wraps

import cf_units
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
from git import Repo
from iris.time import PartialDateTime
from numpy.testing import assert_allclose
from pyhdf.SD import SD, SDC
from tqdm import tqdm

from joblib import Memory
from wildfires.joblib.caching import CodeObj, wrap_decorator
from wildfires.joblib.iris_backend import register_backend

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.expanduser("~"), "FIREDATA")


repo_dir = os.path.join(os.path.dirname(__file__), os.pardir)
repo = Repo(repo_dir)

# Above this mm/h threshold, a day is a 'wet day'.
MM_PER_HR_THRES = 0.1 / 24
M_PER_HR_THRES = MM_PER_HR_THRES / 1000


def data_is_available():
    """Check if DATA_DIR exists.

    Returns:
        bool: True if the data directory exists.

    """
    return os.path.exists(DATA_DIR)


register_backend()
iris_memory = Memory(
    location=DATA_DIR if data_is_available() else None, backend="iris", verbose=0
)


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
        This function realises the cube's lazy data (if any).

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
    for cube in climatology_cubes:
        # This function has the wanted side-effect of realising the data.
        # Without this, calculations of things like the total temporal mean are
        # delayed until the cube is needed, which we do not want here as we are
        # caching the results.
        homogenise_cube_mask(cube)

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


def dummy_lat_lon_cube(data, lat_lims=(-90, 90), lon_lims=(-180, 180), **kwargs):
    """Construct a cube from data given certain assumptions and optional arguments.

    Args:
        lat_lims (2-tuple):
        lon_lims (2-tuple):
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
        new_longitudes, standard_name="longitude", units="degrees"
    )

    if n_dims == 2:
        grid_coords = [(new_lat_coord, 0), (new_lon_coord, 1)]
    else:
        grid_coords = [
            (
                iris.coords.DimCoord(
                    range(data.shape[0]),
                    standard_name="time",
                    var_name="time",
                    units=cf_units.Unit("days since 1970-01-01", calendar="gregorian"),
                ),
                0,
            ),
            (new_lat_coord, 1),
            (new_lon_coord, 2),
        ]

    kwargs_mod = kwargs.copy()
    if "dim_coords_and_dims" in kwargs_mod:
        del kwargs_mod["dim_coords_and_dims"]

    return iris.cube.Cube(
        data,
        dim_coords_and_dims=kwargs.get("dim_coords_and_dims", grid_coords),
        **kwargs_mod,
    )


def data_map_plot(
    data, lat_lims=(-90, 90), lon_lims=(-180, 180), filename=None, log=False, **kwargs
):
    """Used to plot data or an iris.cube.Cube on a map with coastlines.

    """
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


def get_centres(data):
    """Get the elements between elements of an array.

    Examples:
        >>> import numpy as np
        >>> a = np.array([1,2,3])
        >>> b = get_centres(a)
        >>> np.all(np.isclose(b, np.array([1.5, 2.5])))
        True

    """
    return (data[:-1] + data[1:]) / 2.0


def translate_longitudes(lons, sort=True):
    """Go from [-180, 180] to [0, 360] domain."""
    transformed = lons % 360
    if sort:
        assert len(np.unique(np.diff(transformed))) < 3, (
            "Expecting at most 2 unique differences, one for the regular interval, "
            "another for the jump at 0° in case of the [-180, 180] domain."
        )
        transformed = np.sort(transformed)
    return transformed


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
    # NOTE: Execute this step since saving & reloading the cubes containing certain
    # longitudes appears to add the `circular=True` attribute. So to make caching
    # consistent without having to reload data on the first iteration, add this
    # attribute now.
    longitudes.circular = iris.util._is_circular(
        translate_longitudes(longitudes.points), 360
    )
    logger.debug(f"Longitudes are circular: {longitudes.circular}")
    return latitudes, longitudes


def lat_lon_match(
    cube,
    new_latitudes=get_centres(np.linspace(-90, 90, 721)),
    new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
):
    """Test whether regridding is necessary."""
    n_dim = len(cube.shape)
    assert n_dim in {2, 3}, "Need [[time,] lat, lon] dimensions."

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
    area_weighted=False,
    new_latitudes=get_centres(np.linspace(-90, 90, 721)),
    new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
):
    """Keep (optional) time coordinate, but regrid latitudes and longitudes.

    Expects either (time, lat, lon) or (lat, lon) coordinates.

    NOTE: Regarding caching AreaWeighted regridders - the creation of the
    regridder does not seem to take much time, however, so this step is
    almost inconsequential. This is supported by the fact that the source
    code and online bug reports show that no actual caching of weights
    takes place! Furthermore, as time coordinate differences may
    exist between coordinates, iris does not support such differences with
    cached regridders.

    """
    n_dim = len(cube.shape)
    assert n_dim in {2, 3}, "Need [[time,] lat, lon] dimensions."

    # TODO: Check that coordinate system discrepancies are picked up by
    # this check!!
    if lat_lon_match(cube, new_latitudes, new_longitudes):
        logger.info("No regridding needed for '{}'.".format(cube.name()))
        return cube

    logger.debug("Regridding '{}'.".format(cube.name()))

    new_latitudes, new_longitudes = lat_lon_dimcoords(new_latitudes, new_longitudes)

    if n_dim == 3:
        # Call the regridding function recursively with time slices of the
        # data, in order to try to prevent occasional Segmentation Faults
        # that occur when trying to regrid a large chunk of data in 3
        # dimensions.
        regridded_cubes = iris.cube.CubeList()
        assert len(cube.coords()[0].points) == cube.shape[0]
        for i in range(cube.shape[0]):
            regridded_cubes.append(
                regrid(
                    cube[i],
                    area_weighted=area_weighted,
                    new_latitudes=new_latitudes,
                    new_longitudes=new_longitudes,
                )
            )
        return regridded_cubes.merge_cube()

    assert n_dim == 2, "Need [lat, lon] dimensions for core algorithm."

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

    scheme = iris.analysis.AreaWeighted() if area_weighted else iris.analysis.Linear()

    logger.debug("Cube has lazy data: {}.".format(cube.has_lazy_data()))
    logger.debug("Calling regrid with scheme '{}'.".format(scheme))
    interpolated_cube = cube.regrid(new_grid, scheme)

    return interpolated_cube


@dataset_cache
def regrid_dataset(
    dataset,
    area_weighted=False,
    new_latitudes=get_centres(np.linspace(-90, 90, 721)),
    new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
):
    logger.debug(f"Regridding '{dataset}' with {len(dataset)} variable(s).")
    regridded_cubes = iris.cube.CubeList(
        [
            regrid(
                cube,
                area_weighted=area_weighted,
                new_latitudes=new_latitudes,
                new_longitudes=new_longitudes,
            )
            for cube in dataset.cubes
        ]
    )
    return regridded_cubes


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


class Dataset(ABC):
    # TODO: Make sure these get overridden by the subclasses, or that every
    # dataset uses these consistently (if defining custom date coordinates).
    calendar = "gregorian"
    time_unit_str = "days since 1970-01-01 00:00:00"
    time_unit = cf_units.Unit(time_unit_str, calendar="gregorian")

    pretty_variable_names = dict()
    _pretty = None

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
        if isinstance(index, slice):
            new_dataset = self.copy(deep=False)
            new_dataset.cubes = self.cubes[index]
            return new_dataset
        if isinstance(index, str):
            # TODO: Use contains() function to search for pretty names as well, (and to
            # TODO: implement regex matching).
            new_index = self.variable_names(which="raw").index(index)
        else:
            new_index = index
        return self.cubes[new_index]

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
        """Functions that should be run prior to accessing data.

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
            if n_dim == 2:
                coord_names.extend(("latitude", "longitude"))
            elif n_dim == 3:
                coord_names.extend(("time", "latitude", "longitude"))
            else:
                warnings.warn(f"'{cube}' in '{self}' has {n_dim} axes.")

            for coord_name in coord_names:
                try:
                    coord = cube.coord(coord_name)
                    coord.var_name = coord_name
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
            self.__cubes = new_cubes

    @property
    def cube(self):
        """Convenience method to access a single stored cube."""
        if len(self.cubes) != 1:
            raise ValueError(f"Expected 1 cube, but found {len(self.cubes)} cubes.")
        return self.cubes[0]

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

    @property
    def frequency(self):
        try:
            time_coord = self.cubes[0].coord("time")
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
            month_number_coords = self.cubes[0].coords("month_number")
            if month_number_coords:
                assert len(month_number_coords) == 1
                if tuple(month_number_coords[0].points) == tuple(range(1, 13)):
                    return "climatology"
            return "unknown"

        except iris.exceptions.CoordinateNotFoundError:
            return "static"

    @property
    def min_time(self):
        try:
            return self.cubes[0].coord("time").cell(0).point
        except iris.exceptions.CoordinateNotFoundError:
            return "static"

    @property
    def max_time(self):
        try:
            return self.cubes[0].coord("time").cell(-1).point
        except iris.exceptions.CoordinateNotFoundError:
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
            str or None: The current hex commit sha hash of the repo if a
            new file was created. Otherwise, if the file was already there
            and not overwritten, None is returned.

        """
        assert (
            target_filename[-3:] == ".nc"
        ), "Data must be saved as a NetCDF file, got:'{:}'".format(target_filename)
        assert isinstance(cache_data, (iris.cube.Cube, iris.cube.CubeList)), (
            "Data to be saved must either be a Cube or a CubeList. "
            "Got:{:}".format(cache_data)
        )

        if isinstance(cache_data, iris.cube.Cube):
            cache_data = iris.cube.CubeList([cache_data])

        if os.path.isfile(target_filename):
            # TODO: Want to overwrite if the commit hash is different?
            # Maybe add a flag to do this.
            logger.info("File exists, not overwriting:'{:}'".format(target_filename))
        else:
            assert (not repo.untracked_files) and (
                not repo.is_dirty()
            ), "All changes must be committed and all files must be tracked."

            # Note down the commit sha hash so that the code used to
            # generate the cached data can be retrieved easily later on.
            for cube in cache_data:
                cube.attributes["commit"] = repo.head.ref.commit.hexsha

            if not os.path.isdir(os.path.dirname(target_filename)):
                os.makedirs(os.path.dirname(target_filename))
            logger.info("Saving cubes to:'{:}'".format(target_filename))
            iris.save(cache_data, target_filename, zlib=False)
            return cube.attributes["commit"]

    @staticmethod
    def read_data(target_filename):
        """Read from NetCDF file.

        Args:
            target_filename (str): The filename that the data will be saved
                to. Must end in '.nc', since the data is meant to be saved
                as a NetCDF file.

        Raises:
            AssertionError: If the commit hashes of the cubes that are
                loaded do not match.

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
            assert (
                len(set(commit_hashes)) == 1
            ), "Cubes should all stem from the same commit."
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

    def select_data(self, latitude_range=(-90, 90), longitude_range=(-180, 180)):
        self.cube = self.cube.intersection(latitude=latitude_range).intersection(
            longitude=longitude_range
        )
        return self.cube

    def regrid(
        self,
        area_weighted=False,
        new_latitudes=get_centres(np.linspace(-90, 90, 721)),
        new_longitudes=get_centres(np.linspace(-180, 180, 1441)),
    ):
        """Replace stored cubes with regridded versions in-place.

        """
        # The time needed for this check is only on the order of ms.
        if all(lat_lon_match(cube, new_latitudes, new_longitudes) for cube in self):
            logger.info("No regridding needed for '{}'.".format(self.name))
        else:
            for cube_slice in self.single_cube_slices():
                self.cubes[cube_slice] = regrid_dataset(
                    self[cube_slice],
                    area_weighted=area_weighted,
                    new_latitudes=new_latitudes,
                    new_longitudes=new_longitudes,
                )

    @abstractmethod
    def get_monthly_data(self, start, end):
        """Return monthly cubes between two dates."""

    @staticmethod
    def date_order_check(start, end):
        if start is None and end is None:
            return
        if not all(
            hasattr(date, required_type) and getattr(date, required_type) is not None
            for date in (start, end)
            for required_type in ("year", "month")
        ):
            raise ValueError(
                "Both '{}' and '{}' need to define a year and month.".format(start, end)
            )
        days = [getattr(date, "day", 1) for date in (start, end)]
        days = [day if day is not None else 1 for day in days]

        assert datetime(start.year, start.month, days[0]) < datetime(
            end.year, end.month, days[1]
        ), "End date must be greater than start date."

    def limit_months(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        """Discard non-specified time period.

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
        elif freq not in ("monthly",):
            raise ValueError("Invalid frequency:{:}".format(freq))
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

        Daily information is ignored (truncated, ie. days are assumed to be
        1).

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
            Returned cubes may contain lazy data.

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
        self.cubes = iris.cube.CubeList(
            [iris.load_cube(os.path.join(self.dir, "Carvalhais.gpp_50.360.720.1.nc"))]
        )
        # There is only one time coordinate, and its value is of no relevance.
        # Therefore, remove this coordinate.
        self.cubes[0] = self.cubes[0][0]
        self.cubes[0].remove_coord("time")

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.broadcast_static_data(start, end)


class CHELSA(Dataset):
    """For primary analysis, it is advisable to use hpc
    (cx1_scipts/run_chelsa_script.sh) in order to process the tif files
    into nc files as a series of jobs, which would take an incredibly long
    time otherwise (on the order of days).

    Once that script has been run, the resulting nc files can be used to
    easily construct a large iris Cube containing all the data.

    """

    _pretty = "CHELSA"

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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class Copernicus_SWI(Dataset):
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

            for cube in daily_cubes:
                # Make metadata uniform so they can be concatenated.
                del cube.attributes["identifier"]
                del cube.attributes["title"]
                del cube.attributes["time_coverage_start"]
                del cube.attributes["time_coverage_end"]
                del cube.attributes["platform"]
                del cube.attributes["copyright"]
                del cube.attributes["history"]
                del cube.attributes["sensor"]
                del cube.attributes["source"]

            # Concatenate daily cubes into larger cubes with the same
            # information (but with longer time coordinates).
            raw_cubes = daily_cubes.concatenate()

            while raw_cubes:
                logger.debug("Regridding:{:}".format(repr(raw_cubes[0])))
                regridded_cube = regrid(raw_cubes.pop(0))
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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class CRU(Dataset):

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
            self.cubes = iris.load(os.path.join(self.dir, "*.nc"))

        # TODO: For now, remove the 'stn' cubes (see above).
        self.cubes = iris.cube.CubeList(
            [cube for cube in self.cubes if cube.name() != "stn"]
        )

        # Fix units for cloud cover.
        for cube in self.cubes:
            if cube.name() == "cloud cover":
                cube.units = cf_units.Unit("percent")
                break

        # NOTE: Measurement times are listed as being in the middle of the
        # month, requiring no further intervention.

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class ERA5_TotalPrecipitation(Dataset):
    _pretty = "ERA5 Total Precipitation"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ERA5", "tp")
        self.cubes = iris.cube.CubeList(
            [iris.load_cube(os.path.join(self.dir, "*.nc"))]
        )

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class ERA5_DryDayPeriod(Dataset):
    _pretty = "ERA5 Dry Day Period"

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

        self.cubes = iris.cube.CubeList([dry_day_period_cubes.merge_cube()])
        self.write_cache()

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class ERA5_CAPEPrecip(Dataset):
    _pretty = "ERA5 Cape x Precip"

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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


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
    _pretty = "ESA CCI Landcover"

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


class ESA_CCI_Landcover_PFT(Dataset):
    _pretty = "ESA CCI Land Cover PFT"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ESA-CCI-LC_landcover", "0d25_lc2pft")
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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.interpolate_yearly_data(start, end)


class ESA_CCI_Soilmoisture(Dataset):
    _pretty = "ESA CCI Soil Moisture"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "ESA-CCI-SM_soilmoisture")
        self.cubes = iris.load(os.path.join(self.dir, "*.nc"))

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class ESA_CCI_Soilmoisture_Daily(Dataset):
    _pretty = "ESA CCI Daily Soil Moisture"

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


class GFEDv4(Dataset):
    """Without small fires.

    """

    _pretty = "GFED4"

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

        # Normalise using the areas, divide by 10000 to convert from m2 to
        # hectares (the burned areas are in hectares originally).
        # NOTE: Some burned area percentages may be above 1!
        burned_area_cube.data /= (
            iris.analysis.cartography.area_weights(burned_area_cube) / 10000
        )
        burned_area_cube.units = cf_units.Unit("percent")

        self.cubes = iris.cube.CubeList([burned_area_cube])
        self.write_cache()

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class GFEDv4s(Dataset):
    """Includes small fires.

    """

    _pretty = "GFED4s"

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
            container = h5py.File(f)

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

        self.cubes[0].units = cf_units.Unit("percent")
        self.cubes[0].var_name = "Burnt_Area"
        self.write_cache()

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class GlobFluo_SIF(Dataset):
    _pretty = "Glob Fluo SIF"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "GlobFluo_SIF")
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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


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
        """Linear interpolation onto the target months.

        """
        final_cubelist = self.interpolate_yearly_data(start, end)
        assert len(final_cubelist) == 1
        return final_cubelist


class GSMaP_dry_day_period(Dataset):
    """Calculate the length of the longest preceding dry day period.

    This definition only considers dry day periods within the current month, or dry
    day periods that occur within the current month AND previous months, ONLY if these
    join up contiguously at the month boundaries.

    Other definitions taking into account (only) dry day periods in a certain number
    of months leading up to the current month may be possible as well, although this
    could also be implemented in post-processing.

    """

    _pretty = "GSMaP Dry Day Period"

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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class GSMaP_precipitation(Dataset):
    _pretty = "GSMaP Precipitation"

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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


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
        """Linear interpolation onto the target months.

        """
        return self.interpolate_yearly_data(start, end)


class LIS_OTD_lightning_climatology(Dataset):
    _pretty = "LIS/OTD Climatology"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "LIS_OTD_lightning_climatology")
        self.cubes = iris.cube.CubeList(
            [
                iris.load(os.path.join(self.dir, "*.nc")).extract_strict(
                    iris.Constraint(name="Combined Flash Rate Monthly Climatology")
                )
            ]
        )
        # Fix time units so they do not refer months, as this can't be processed by
        # iris / cf_units.
        # Realign times so they are at the centre of each month.

        # Check that existing time coordinate is as expected.
        assert self.cubes[0].coord("time").units.origin == "months since 2014-1-1 0:0:0"
        assert all(self.cubes[0].coord("time").points == np.arange(1, 13))

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
        self.cubes[0].coord("time").points = time_coord.points
        self.cubes[0].coord("time").units = time_coord.units

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

        NOTE: Not 'True' monthly data!!

        This method ignores days.

        """
        # TODO: Make this work with lazy data?

        cube = self.cubes[0]
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


class LIS_OTD_lightning_time_series(Dataset):
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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class Liu_VOD(Dataset):
    _pretty = "Liu VOD"

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

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        return self.select_monthly_from_monthly(start, end)


class MOD15A2H_LAI_fPAR(Dataset):
    _pretty = "MOD15A2H"

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, "MOD15A2H_LAI-fPAR")
        self.cubes = iris.load(os.path.join(self.dir, "*.nc"))

        months = []
        for i in range(self.cubes[0].shape[0]):
            months.append(self.cubes[0].coords()[0].cell(i).point.month)

        assert np.all(
            np.diff(np.where(np.diff(months) != 1)) == 12
        ), "The year should increase every 12 samples!"

    def get_monthly_data(
        self, start=PartialDateTime(2000, 1), end=PartialDateTime(2000, 12)
    ):
        # TODO: Since the day in the month for which the data is provided
        # is variable, take into account neighbouring months as well in a
        # weighted average (depending on how many days away from the middle
        # of the month these other samples are)?
        return self.select_monthly_from_monthly(start, end)


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


def dataset_times(datasets=None, dataset_names=None):
    """Compile dataset time span information.

    Args:
        datasets (iterable of `Dataset`): If no value is given, defaults to using the
            following datasets:
                - AvitabileThurnerAGB(),
                - CHELSA(),
                - Copernicus_SWI(),
                - ESA_CCI_Landcover_PFT(),
                - GFEDv4(),
                - GSMaP_precipitation(),
                - GlobFluo_SIF(),
                - HYDE(),
                - LIS_OTD_lightning_time_series(),
                - Liu_VOD(),
                - MOD15A2H_LAI_fPAR(),
                - Simard_canopyheight(),
                - Thurner_AGB(),
            Alternatively, a list of Dataset instances can be given.
        dataset_names (iterable of `str` or None): The names used for the datasets,
            the number of which should match the number of items in `datasets`. If
            None, use the `dataset.pretty` name for each `Dataset` in `datasets`.

    Returns:
        If valid starting and end times can be found for at least one of the datasets:
            - min_time: Minimum shared time of all datasets.
            - max_time: Maximum shared time of all datasets.
            - times_df: Pandas DataFrame encapsulating the timespan information.
        Otherwise the 3-tuple (None, None, None) will be returned.

    """
    # TODO: Use the 'get_all_datasets' function from cube_aggregation.py here, listing
    # the excluded datasets only in the docstring. This should make accommodating new
    # datasets even easier.
    if datasets is None:
        datasets = [
            AvitabileThurnerAGB(),
            CHELSA(),
            Copernicus_SWI(),
            ESA_CCI_Landcover_PFT(),
            GFEDv4(),
            GSMaP_precipitation(),
            GlobFluo_SIF(),
            HYDE(),
            LIS_OTD_lightning_time_series(),
            Liu_VOD(),
            MOD15A2H_LAI_fPAR(),
            Simard_canopyheight(),
            Thurner_AGB(),
        ]

    if dataset_names is None:
        dataset_names = tuple(dataset.pretty for dataset in datasets)

    time_dict = OrderedDict(
        (name, list(map(str, (dataset.min_time, dataset.max_time, dataset.frequency))))
        for dataset, name in zip(datasets, dataset_names)
    )
    min_times, max_times = [], []
    for dataset in datasets:
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

    time_dict["Overall"] = list(map(str, (min_time, max_time, "N/A")))

    dataset_names = list(time_dict.keys())
    dataset_names = pd.Series(dataset_names, name="Dataset")
    min_times_series = pd.Series(
        [time_dict[name][0] for name in dataset_names], name="Minimum"
    )
    max_times_series = pd.Series(
        [time_dict[name][1] for name in dataset_names], name="Maximum"
    )
    frequency_series = pd.Series(
        [time_dict[name][2] for name in dataset_names], name="Frequency"
    )
    times_df = pd.DataFrame(
        [dataset_names, min_times_series, max_times_series, frequency_series]
    ).T

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
