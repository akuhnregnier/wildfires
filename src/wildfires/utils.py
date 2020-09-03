# -*- coding: utf-8 -*-
"""Collection of code to be used throughout the project.

"""
import logging
import math
import os
import pickle
import re
import shlex
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter
from copy import deepcopy
from functools import partial, wraps
from pathlib import Path
from pickle import UnpicklingError
from subprocess import check_output
from tempfile import NamedTemporaryFile
from textwrap import dedent
from time import time
from warnings import warn

import fiona
import iris
import numpy as np
from affine import Affine
from rasterio import features
from scipy.ndimage import label
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NoCachedDataError(Exception):
    """Raised when the cache pickle file could not be found."""


class CoordinateSystemError(Exception):
    """Raised when an unknown coordinate system is encountered."""


class SimpleCache:
    """Simple caching functionality without analysing arguments."""

    def __init__(self, filename, cache_dir=".pickle", verbose=10, pickler=pickle):
        """Initialise the cacher.

        Args:
            filename (str): Name of the file to save to.
            cache_dir (str): Directory `filename` will be created in.
            verbose (int): If `verbose >= 10`, logging messages will be printed to stdout.
            pickler (object): An object with 'load' and 'dump' methods analogous to pickle.

        """
        os.makedirs(cache_dir, exist_ok=True)
        self.pickle_path = os.path.join(cache_dir, filename)
        self.verbose = verbose
        self.pickler = pickler

    def available(self):
        """Check if data has been cached."""
        avail = os.path.isfile(self.pickle_path)
        if self.verbose >= 10:
            if avail:
                print(f"Data found at {self.pickle_path}.")
            else:
                print(f"Data not found at {self.pickle_path}.")
        return avail

    def load(self):
        """Load cached data.

        Returns:
            Loaded data.

        Raises:
            NoCachedDataError: If no cached data was found.

        """
        if self.available():
            try:
                with open(self.pickle_path, "rb") as f:
                    return self.pickler.load(f)
            except UnpicklingError:
                logger.warning(f"Data at '{self.pickle_path}' could not be loaded.")
                raise NoCachedDataError(f"{self.pickle_path} contained corrupted data.")
        raise NoCachedDataError(f"{self.pickle_path} does not exist.")

    def save(self, obj):
        """Cache `obj`."""
        if self.verbose >= 10:
            print(f"Saving data to {self.pickle_path}.")
        with open(self.pickle_path, "wb") as f:
            self.pickler.dump(obj, f, -1)

    def clear(self):
        """Delete cached contents (if any)."""
        if self.verbose >= 10:
            print(f"Clearing data from {self.pickle_path}.")
        if os.path.isfile(self.pickle_path):
            os.remove(self.pickle_path)

    def __call__(self, func):
        """Simple caching decorator."""

        @wraps(func)
        def cached_func(*args, **kwargs):
            if args or kwargs:
                warn(
                    "Parameters are not considered when saving/loading cached results."
                )
            try:
                return self.load()
            except NoCachedDataError:
                if self.verbose >= 10:
                    print(f"Calling {func}.")
                start = time()
                results = func(*args, **kwargs)
                eval_time = time() - start
                if self.verbose >= 10:
                    print(f"Finished call. Time taken: {self.float_format(eval_time)}s")
                self.save(results)
                save_time = time() - eval_time - start
                if self.verbose >= 10:
                    print(
                        f"Finished saving. Time taken: {self.float_format(save_time)}s"
                    )
                return results

        return cached_func

    def __repr__(self):
        return f"SimpleCache at {self.pickle_path} - saved data: {self.available()}."

    def __str__(self):
        return repr(self)

    @staticmethod
    def float_format(number, additional=0):
        """Float formatting that only retains decimal places for small numbers.

        Args:
            number (float): Number to format.
            additional (int): Number of additional decimal places to use.

        Returns:
            str: Formatted `number`.

        """
        if number < 10:
            dec = math.ceil(abs(math.log10(number)))
            if number <= 1:
                dec += 1
        else:
            dec = 0
        return f"{number:0.{dec + additional}f}"


class TqdmContext(tqdm):
    """Use like:
        `with TqdmContext(unit=" plots", desc="Plotting", total=10) as t:`

    Where `total` refers to the total number of elements.

    Call t.update_to(iteration) which will increment the internal counter to
    `iteration`.

    Add the total keyword to change the total number of expected iterations.

    Alternatively, call t.update() (defined in the core tqdm class) to increment the
    counter by 1.

    """

    def update_to(self, total=None):
        if total is not None:
            self.total = total
        self.update()


class Time:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        print("Time taken for {}: {}".format(self.name, time() - self.start))


class RampVar:
    """Variable that is increased upon every call.

    The starting value, maximum value and the steps can be set.

    The value is incremented linearly between the initial and maximum
    value, with `steps` intervals.

    Args:
        initial_value (float): Initial value.
        max_value (float): Maximum value the variable can take.
        steps (int): The number of intervals.

    Examples:
        >>> var = RampVar(0, 2, 3)
        >>> int(round(var.value))
        0
        >>> int(round(var.value))
        1
        >>> int(round(var.value))
        2
        >>> var.reset()
        >>> int(round(var.value))
        0

    """

    def __init__(self, initial_value, max_value, steps=10):
        self.steps = steps
        self.values = np.linspace(initial_value, max_value, steps)
        self.index = -1

    @property
    def value(self):
        """Every time this attribute is accessed it is incremented as
        defined by the values given to the constructor.

        """
        if self.index < self.steps - 1:
            self.index += 1
        return self.values[self.index]

    def reset(self):
        """Resets the value to the initial value."""
        self.index = -1


def get_land_mask(n_lon=1440, ignore_indices=(7, 126)):
    """Create land mask at the desired resolution.

    Data is taken from https://www.naturalearthdata.com/

    Args:
        n_lon (int): The number of longitude points of the final mask array. As the
            ratio between number of longitudes and latitudes has to be 2 in order for
            uniform scaling to work, the number of latitudes points is calculated as
            n_lon / 2.
        ignore_indices (iterable of int or None): Ignore geometries with indices in
            `ignore_indices` when constructing the mask. Indices (7, 126) refer to
            Antarctica and Greenland respectively.

    Returns:
        numpy.ndarray: Array of shape (n_lon / 2, n_lon) and dtype np.bool_. True
            where there is land, False otherwise.

    Examples:
        >>> import numpy as np
        >>> from wildfires.data.datasets import data_is_available
        >>> if data_is_available():
        ...     mask = get_land_mask(n_lon=1440)
        ...     assert mask.dtype == np.bool_
        ...     assert mask.shape == (720, 1440)

    """
    from wildfires.data.datasets import DATA_DIR

    assert n_lon % 2 == 0, (
        "The number of longitude points has to be an even number for the number of "
        "latitude points to be an integer."
    )
    n_lat = round(n_lon / 2)
    geom_np = np.zeros((n_lat, n_lon), dtype=np.uint8)
    with fiona.open(
        os.path.join(DATA_DIR, "land_mask", "ne_110m_land.shp"), "r"
    ) as shapefile:
        for i, geom in enumerate(shapefile):
            if ignore_indices and i in ignore_indices:
                continue
            geom_np += features.rasterize(
                [geom["geometry"]],
                out_shape=geom_np.shape,
                dtype=np.uint8,
                transform=~(
                    Affine.translation(n_lat, n_lat / 2) * Affine.scale(n_lon / 360)
                ),
            )

    geom_np = geom_np.astype(np.bool_)
    return geom_np


def polygon_mask(coordinates, n_lon=1440):
    """Mask based on a rasterized polygon from specified coordinates.

    Args:
        coordinates (list of tuple of float): List of (longitude, latitude)
            coordinates specified in either clockwise or anti-clockwise order. The
            last point MUST be the same as the first point for the polygon to be
            recognised as a closed, valid shape. Longitudes are specified in the
            interval [-180, 180], and latitudes in the interval [-90, 90].
        n_lon (int): The number of longitude points of the final mask array. As the
            ratio between number of longitudes and latitudes has to be 2 in order for
            uniform scaling to work, the number of latitudes points is calculated as
            n_lon / 2.

    Returns:
        numpy.ndarray: Array of shape (n_lon / 2, n_lon) and dtype np.bool_. True
        inside the specified polygon, False otherwise.

    Examples:
        >>> import numpy as np
        >>> data = np.arange(720*1440).reshape(720, 1440)
        >>> # Mask the lower half of the globe.
        >>> data[
        ...     polygon_mask([(180, -90), (-180, -90), (-180, 0), (180, 0), (180, -90)])
        ... ] = 0
        >>> np.isclose(data.mean(), 388799.75)
        True

    """
    assert n_lon % 2 == 0, (
        "The number of longitude points has to be an even number for the number of "
        "latitude points to be an integer."
    )
    n_lat = round(n_lon / 2)
    geom_np = np.zeros((n_lat, n_lon), dtype=np.uint8)
    geom_np += features.rasterize(
        [{"type": "Polygon", "coordinates": [coordinates]}],
        out_shape=geom_np.shape,
        dtype=np.uint8,
        transform=~(Affine.translation(n_lat, n_lat / 2) * Affine.scale(n_lon / 360)),
    )

    geom_np = geom_np.astype(np.bool_)
    return geom_np


def box_mask(lats, lons, n_lon=1440):
    """Mask based on a rasterized box from specified coordinates.

    Args:
        lats (2-iterable of float): Minimum and maximum latitudes. Latitudes are
            specified in the interval  [-90, 90].
        lons (2-iterable of float): Minimum and maximum latitudes. Longitudes are
            specified in the interval [-180, 180].
        n_lon (int): The number of longitude points of the final mask array. As the
            ratio between number of longitudes and latitudes has to be 2 in order for
            uniform scaling to work, the number of latitudes points is calculated as
            n_lon / 2.

    Returns:
        numpy.ndarray: Array of shape (n_lon / 2, n_lon) and dtype np.bool_. True
        inside the specified limits, False otherwise.

    """
    # Go around the box clockwise.
    coordinates = [
        (lons[0], lats[0]),
        (lons[1], lats[0]),
        (lons[1], lats[1]),
        (lons[0], lats[1]),
    ]
    # Make sure the last point matches the first point.
    coordinates.append(coordinates[0])
    return polygon_mask(coordinates, n_lon=n_lon)


def pack_input(var, single_type=str, elements=2, fill_source=0):
    """Return a filled tuple with `elements` items.

    Args:
        var (iterable of `single_type` or `single_type`): Input variable which
            will be transformed.
        single_type (class or tuple of class): Atomic type(s) that will be treated
            as single items.
        elements (int): Number of elements in the final tuple.
        fill_source (int, None, or iterable of int or None): Determines how to pad
            input such that it contains `elements` items. No existing items in
            `var will be altered. If `fill_source` is an int or None, it is
            treated as an iterable containing `fill_source` (`elements` -
            len(`var`)) times, where len(`var`) refers to the number of
            `single_type` items supplied in `var` (which may be only one, in which
            case `var` is internally transformed to be a 1-element iterable
            containing `var`). The `fill_source` iterable must contain at least
            (`elements` - len(`var`)) items, since this is the number of slots
            that need to be filled in order for the output to contain `elements`
            items. If `fill_source[-i]` is an int, `output[-i]` will be inserted into
            `output` at index `elements - i`. If `fill_source[-i]` is None, None
            will be inserted. Surplus `fill_source` items will be trimmed starting
            from the left (thus the -i index notation above).

    Returns:
        tuple: tuple with `elements` items.

    Raises:
        ValueError: If `var` is an iterable of `single_type` and contains more than
            `elements` items.
        TypeError: If `var` is an iterable and its items are not all of type
            `single_type`.
        TypeError: If `fill_source` contains types other than int and NoneType.
        IndexError: If `len(fill_source)` < (`elements` - len(`var`)).
        IndexError: If `fill_source[-i]` is an int and
            `fill_source[-i]` >= `elements` - i.

    Examples:
        >>> pack_input("testing")
        ('testing', 'testing')
        >>> pack_input(("foo",))
        ('foo', 'foo')
        >>> pack_input(("foo", "bar"), elements=3, fill_source=1)
        ('foo', 'bar', 'bar')
        >>> pack_input("foo", elements=2, fill_source=None)
        ('foo', None)
        >>> pack_input("foo", elements=3, fill_source=(0, None))
        ('foo', 'foo', None)
        >>> # Surplus `fill_source` items will be trimmed starting from the left.
        >>> pack_input("foo", elements=3, fill_source=(99, 0, None))
        ('foo', 'foo', None)
        >>> pack_input(("foo", "bar"), elements=5, fill_source=(1, 2, None))
        ('foo', 'bar', 'bar', 'bar', None)

    """
    if not isinstance(var, single_type):
        if not all(isinstance(single_var, single_type) for single_var in var):
            raise TypeError(
                "Expected items to be of type(s) '{}', but got types '{}'.".format(
                    single_type, [type(single_var) for single_var in var]
                )
            )
        if len(var) > elements:
            raise ValueError(
                "Expected at most {} item(s), got {}.".format(elements, len(var))
            )
        if len(var) == elements:
            return tuple(var)
        # Guarantee that `var` is a list, and make a copy so the input is not
        # changed unintentionally.
        var = list(var)
    else:
        var = [var]

    fill_source_types = (int, type(None))
    if not isinstance(fill_source, fill_source_types):
        if not all(
            isinstance(single_source, fill_source_types)
            for single_source in fill_source
        ):
            raise TypeError(
                "Expected fill_source to be of types '{}', but got types '{}'.".format(
                    fill_source_types,
                    [type(single_source) for single_source in fill_source],
                )
            )
        # Again, make a copy.
        fill_source = fill_source[:]
    else:
        fill_source = [fill_source] * (elements - len(var))

    n_missing = elements - len(var)
    for i in range(-n_missing, 0):
        if fill_source[i] is None:
            fill_value = None
        else:
            fill_value = var[fill_source[i]]
        var.append(fill_value)
    return tuple(var)


def match_shape(array, target_shape):
    """Broadcast an array across the first axis.

    A new axis will be inserted at the beginning if needed.

    Args:
        array (numpy.ndarray): Numpy array with either 2 or 3 dimensions.
        target_shape (tuple of int): Target shape.

    Returns:
        numpy.ndarray: Boolean array with shape `target_shape`.

    Examples:
        >>> import numpy as np
        >>> mask = np.zeros((4, 4), dtype=np.bool_)
        >>> match_shape(mask, (10, 4, 4)).shape
        (10, 4, 4)
        >>> mask = np.zeros((1, 4, 4), dtype=np.bool_)
        >>> match_shape(mask, (10, 4, 4)).shape
        (10, 4, 4)
        >>> mask = np.zeros((10, 4, 4), dtype=np.bool_)
        >>> match_shape(mask, (10, 4, 4)).shape
        (10, 4, 4)
        >>> mask = np.array([1, 0, 1], dtype=np.bool_)
        >>> np.all(
        ...     match_shape(mask, (2, 3))
        ...     == np.array([[1, 0, 1], [1, 0, 1]], dtype=np.bool_)
        ...     )
        True

    """
    if array.shape != target_shape:
        # Remove singular first dimension.
        if len(array.shape) == len(target_shape):
            if array.shape[0] == 1:
                array = array[0]
        if array.shape == target_shape[1:]:
            logger.debug(
                "Adding time dimension ({}) to broadcast array.".format(target_shape[0])
            )
            new_array = np.zeros(target_shape, dtype=np.bool_)
            new_array += array.reshape(1, *array.shape)
            array = new_array
        else:
            raise ValueError(
                "Array dimensions '{}' do not match cube dimensions '{}'.".format(
                    array.shape, target_shape
                )
            )
    return array


def get_unmasked(array, strict=True):
    """Get the flattened unmasked elements from a masked array.

    Args:
        array (numpy.ma.core.MaskedArray or numpy.ndarray): If `strict` (default),
            only accept masked arrays.
        strict (bool): See above.

    Returns:
        numpy.ndarray: Flattened, unmasked data.

    Raises:
        TypeError: If `strict` and `array` is of type `numpy.ndarray`. Regardless of
            `strict`, types other than `numpy.ma.core.MaskedArray` and `numpy.ndarray`
            will also raise a TypeError.

    """
    accepted_types = [np.ma.core.MaskedArray]
    if not strict:
        accepted_types.append(np.ndarray)

    if not isinstance(array, tuple(accepted_types)):
        raise TypeError(f"The input array had an invalid type '{type(array)}'.")

    if not strict and isinstance(array, np.ndarray):
        return array.ravel()

    if isinstance(array.mask, np.ndarray):
        return array.data[~array.mask].ravel()
    elif array.mask:
        np.array([])
    else:
        return array.ravel()


def get_masked_array(data, mask=False, dtype=np.float64):
    """Get a masked array from data and an optional mask.

    Args:
        data (iterable):
        mask (numpy.ndarray or bool):
        dtype (numpy dtype):

    Returns:
        numpy.ma.core.MaskedArray

    Examples:
        >>> import numpy as np
        >>> print(get_masked_array([1, 2], [True, False, False], np.int64))
        [-- 1 2]
        >>> print(get_masked_array([0, 1, 2], [True, False, False], np.int64))
        [-- 1 2]
        >>> print(get_masked_array([0, 1, 2], dtype=np.int64))
        [0 1 2]
        >>> a = np.arange(20).reshape(5, 4)
        >>> b = np.arange(7*4).reshape(7, 4)
        >>> mask = np.zeros((7, 4), dtype=np.bool_)
        >>> mask[np.logical_or(b < 4, b > 23)] = True
        >>> stacked = np.vstack((np.zeros((1, 4)), a, np.zeros((1, 4))))
        >>> ma = np.ma.MaskedArray(stacked, mask=mask)
        >>> np.all(ma == get_masked_array(a, mask, np.int64))
        True

    """
    data = np.asarray(data)
    mask = np.asarray(mask, dtype=np.bool_)
    # Make sure mask is an array and not just a single value, and that the data and
    # mask sizes differ.
    if mask.shape and data.size != mask.size:
        shape = mask.shape
        array_data = np.zeros(shape, dtype=dtype).ravel()
        array_data[~mask.ravel()] = data.ravel()
        array_data = array_data.reshape(shape)
        return np.ma.MaskedArray(array_data, mask=mask)
    return np.ma.MaskedArray(data, mask=mask, dtype=dtype)


def in_360_longitude_system(longitudes, tol=1e-4):
    """Determine if the longitudes are represented in the [0, 360] system.

    Note: `np.all` seems to have issues with tolerances lower than ~1e-5.

    Args:
        longitudes (1-D iterable): Longitudes to translate.
        tol (float): Floating point tolerance.

    Returns:
        bool: True if `longitudes` are in [0, 360], False otherwise.

    Raises:
        CoordinateSystemError: If none of the intervals [-180, 180] or [0, 360] match
            `longitudes`.

    Examples:
        >>> in_360_longitude_system([0, 180, 360])
        True
        >>> in_360_longitude_system([0, 180])
        False
        >>> in_360_longitude_system([-180, 0, 180])
        False

    """
    longitudes = np.asarray(longitudes)
    if np.any(longitudes < (-180 - tol)):
        raise CoordinateSystemError("Longitudes below -180 were found.")
    if np.any(longitudes > (360 + tol)):
        raise CoordinateSystemError("Longitudes above 360 were found.")
    if np.any(longitudes > 180):
        if np.any(longitudes < -tol):
            raise CoordinateSystemError(
                "If longitudes over 180 are present, there should be no "
                "longitudes below 0."
            )
        return True
    return False


def translate_longitude_system(longitudes, return_indices=False):
    """Translate the longitudes from one system to another.

    Note:
        The resulting longitudes may not be returned in the initial order, see
        `return_indices`.

    Args:
        longitudes (1-D iterable): Longitudes to translate.
        return_indices (bool): Return the indices used in the post-translation sort.
            These can be used to translate the corresponding datasets using
            `numpy.take` for example.

    Returns:
        translated, sorted longitudes [, argsort indices].

    Examples:
        >>> list(translate_longitude_system([-180, -179, -90, -1, 0, 180]))
        [0, 180, 180, 181, 270, 359]
        >>> # Take care with the extrema! Notice that the above input is not restored
        >>> # below due to the asymmetric mapping [-180, 180] to [0, 360) vs [0, 360]
        >>> # to [-180, 180).
        >>> list(translate_longitude_system([0, 180, 180, 181, 270, 359]))
        [-180, -180, -179, -90, -1, 0]
        >>> list(translate_longitude_system([0, 180, 270, 359, 360]))
        [-180, -90, -1, 0, 0]

    """
    if in_360_longitude_system(longitudes):
        new_longitudes = ((np.asarray(longitudes) + 180) % 360) - 180
    else:
        new_longitudes = np.asarray(longitudes) % 360

    indices = np.argsort(new_longitudes)
    if return_indices:
        return new_longitudes[indices], indices
    else:
        return new_longitudes[indices]


def reorder_cube_coord(cube, indices, new_coord_points, axis):
    """Use indices and the corresponding axis to reorder a cube's data along that axis.

    Args:
        cube (iris.cube.Cube): Cube to be modified.
        indices (1-D iterable): Indices used to select new ordering of values along
            the chosen axis.
        new_coord_points (1-D iterable): New coordinates along `axis`. The length of
            this iterable needs to match the number of indices
            (`len(new_coord_points) == len(indices)`).
        axis (int): Axis along which to reorder the cube.

    Returns:
        iris.cube.Cube: Reordered cube.

    Examples:
        >>> from wildfires.data.datasets import dummy_lat_lon_cube
        >>> import numpy as np
        >>> data = np.arange(4).reshape(2, 2)
        >>> a = dummy_lat_lon_cube(data)
        >>> old_lon_points = a.coord("longitude").points
        >>> indices = [1, 0]
        >>> tr_longitudes = old_lon_points[indices]
        >>> b = reorder_cube_coord(a, indices, tr_longitudes, 1)
        >>> np.all(np.isclose(b.data, data[:, ::-1]))
        True
        >>> id(a) != id(b)
        True
        >>> b2 = reorder_cube_coord(a, indices, tr_longitudes, -1)
        >>> np.all(np.isclose(b2.data, data[:, ::-1]))
        True
        >>> new_lon = [90, -90]
        >>> np.all(np.isclose(b.coord("longitude").points, new_lon))
        True

    """
    indices = np.asarray(indices)
    tr_new_data = np.take(cube.data, indices, axis=axis)

    # Cut down to the required size, then substitute the reordered data.
    slices = [slice(None)] * len(cube.shape)
    slices[axis] = slice(indices.size)
    # XXX: Can we not modify the cube in-place?
    new_cube = deepcopy(cube)[tuple(slices)]
    new_cube.data = tr_new_data

    coord_names = [coord.name() for coord in cube.coords()]
    assert len(coord_names) == len(
        cube.shape
    ), "Each dimension should be associated with a coordinate."
    coord_name = coord_names[axis]
    new_lon = new_cube.coord(coord_name)
    new_lon.points = new_coord_points
    new_lon.bounds = None

    if cube.coord(coord_name).has_bounds():
        new_lon.guess_bounds()
    return new_cube


def select_valid_subset(data, axis=None, longitudes=None):
    """Extract contiguous subset of `data` by removing masked borders.

    Args:
        data (numpy.ma.core.MaskedArray or iris.cube.Cube): Data needs to have an
            array mask.
        axis (int, tuple of int, or None): Axes to subject to selection. If `None`,
            all axes will be considered.
        longitudes (1-D iterable): Longitudes associated with the last axis of the
            data, ie. `len(longitudes) == data.shape[-1]`. If given, they will be
            assumed to be circular (although this isn't checked explicitly) and the
            corresponding (last) axis will be shifted (rolled) to achieve the most
            dense data representation, eliminating the single biggest gap possible.
            Gaps are determined by the `data` mask (`data.mask` or `data.data.mask`
            for an iris Cube). If longitudes are supplied, both the (transformed - if
            needed) data and longitudes will be returned.

    Returns:
        translated (array-like): (Translated) subset of `data`.
        translated_longitudes (array-like): (Translated) longitudes, present only if
            `longitudes` it not None.

    Examples:
        >>> import numpy as np
        >>> np.all(
        ...     np.isclose(
        ...         select_valid_subset(
        ...             np.ma.MaskedArray([1, 2, 3, 4], mask=[1, 1, 0, 0])
        ...         ),
        ...         [3, 4],
        ...     )
        ... )
        True

    """
    if isinstance(data, iris.cube.Cube):
        mask = data.data.mask
    else:
        mask = data.mask

    if isinstance(mask, (bool, np.bool_)):
        raise ValueError(f"Mask is '{mask}'. Expected an array instead.")

    all_axes = tuple(range(len(data.shape)))

    if axis is None:
        axis = all_axes
    elif isinstance(axis, (int, np.integer)):
        axis = (axis,)
    elif not isinstance(axis, tuple):
        raise ValueError(f"Invalid axis ('{axis}') type '{type(axis)}'.")

    slices = [slice(None)] * len(data.shape)
    lon_ax = all_axes[-1]

    # Determine if longitude translation is possible and requested.
    attempt_translation = False
    if longitudes is not None:
        if len(longitudes) != data.shape[lon_ax]:
            raise ValueError(
                "The number of longitudes should match the last data dimension."
            )
        if lon_ax in axis:
            # If longitudes are to be shifted, do not trim this axis, as the number of
            # masked elements are what we are interested in.
            axis = tuple(x for x in axis if x != lon_ax)
            attempt_translation = True

    # Check how much the original data could be compressed by ignoring elements along
    # each of the `axis` boundaries. If longitude translation should be attempted
    # later, the longitude axis is exempt from this (see `axis` definition above).
    for ax in axis:
        # Compress mask such that only the axis of interest remains.
        compressed = np.all(mask, axis=tuple(x for x in all_axes if x != ax))

        elements, n_elements = label(compressed)
        if not n_elements:
            # If no masked elements were found there is nothing to do.
            continue
        ini_index = 0
        fin_index = data.shape[ax]

        # Check the beginning.
        if elements[0]:
            # True (masked) elements are clustered with labels > 0.
            # Count how many elements belong to this feature.
            ini_index += np.sum(elements == elements[0])

        # Ditto or the end.
        if elements[-1]:
            fin_index -= np.sum(elements == elements[-1])

        slices[ax] = slice(ini_index, fin_index)

    # Eliminate data along non-longitude axes first, since we are only allowed to
    # remove one block from the longitudes (the largest block) in order to maintain
    # continuity.
    data = data[tuple(slices)]

    # Compress the mask so only the longitude axis remains.
    non_lon_axis = tuple(x for x in all_axes if x != lon_ax)
    compressed = np.all(mask, axis=non_lon_axis)
    elements, n_elements = label(compressed)

    lon_slice = slice(None)
    lon_slices = [slice(None)] * len(data.shape)
    if n_elements:
        # Find the largest contiguous invalid block.
        invalid_counts = Counter(elements[elements != 0])
        logger.debug(f"Invalid longitude clusters: {invalid_counts}.")

        largest_cluster = max(invalid_counts, key=invalid_counts.__getitem__)

        initial_cut = invalid_counts.get(elements[0], 0)
        final_cut = invalid_counts.get(elements[-1], 0)

        if (initial_cut + final_cut) >= invalid_counts[largest_cluster]:
            # If we can already remove the most elements now there is no point
            # shifting longitudes later.
            attempt_translation = False
            lon_slice = slice(initial_cut, data.shape[lon_ax] - final_cut)
            lon_slices[lon_ax] = lon_slice
    else:
        logger.debug("No invalid longitude clusters were found.")

    if not attempt_translation or not n_elements:
        # If we cannot shift the longitudes, or if no masked elements were found, then
        # there is nothing left to do.
        if longitudes is not None:
            return data[tuple(lon_slices)], longitudes[lon_slice]
        return data

    logger.info("Carrying out longitude translation.")

    # Try shifting longitudes to remove masked elements. The goal is to move the
    # largest contiguous block of invalid elements along the longitude axis to the end
    # of the axis where it can then be sliced off.
    last_cluster_index = np.where(elements == largest_cluster)[0][-1]

    # Shift all data along the longitude axis such that `last_cluster_index` is last.
    shift_delta = data.shape[lon_ax] - last_cluster_index - 1

    logger.debug(f"Shifting longitudes by: {shift_delta} indices.")

    # Create original indices.
    indices = np.arange(data.shape[lon_ax], dtype=np.int64)
    # Translate the data forwards (by subtracting the desired number of shifts).
    indices -= shift_delta
    # Make sure indices wrap around.
    indices %= data.shape[lon_ax]

    # Having shifted the indices, remove the invalid indices which are now at the end.
    indices = indices[: -invalid_counts[largest_cluster]]

    shifted_longitudes = np.take(longitudes, indices)

    # Remove the longitude coordinate discontinuity introduced by the shift.
    shifted_longitudes[shift_delta:] += 360

    if not iris.util.monotonic(shifted_longitudes, strict=True):
        # We need to transform longitudes to be monotonic.
        logger.debug("Translating longitude system.")
        tr_longitudes, transform_indices = translate_longitude_system(
            shifted_longitudes, return_indices=True
        )
        tr_indices = np.take(indices, transform_indices)
    else:
        tr_longitudes = shifted_longitudes
        tr_indices = indices

    # Translate the data and longitudes using the indices.
    if isinstance(data, iris.cube.Cube):
        # The `take` operation is not compatible with iris Cubes.
        new_data = reorder_cube_coord(
            data, tr_indices, new_coord_points=tr_longitudes, axis=lon_ax
        )
    else:
        new_data = np.take(data, tr_indices, axis=lon_ax)

    return new_data, tr_longitudes


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


def get_bounds_from_centres(data):
    """Get coordinate bounds from a series of cell centres.

    Only the centre extrema are considered and an equal spacing between samples is
    assumed.

    Args:
        data (array-like): Cell centres, which will be processed along axis 0.

    Returns:
        array-like: (min, max) coordinate bounds.

    Examples:
        >>> import numpy as np
        >>> centres = [0.5, 1.5, 2.5]
        >>> np.all(
        ...     np.isclose(
        ...         get_bounds_from_centres(centres),
        ...         [0.0, 3.0]
        ...     )
        ... )
        True

    """
    data_min = np.min(data)
    data_max = np.max(data)
    half_spacing = (data_max - data_min) / (2 * (len(data) - 1))
    return data_min - half_spacing, data_max + half_spacing


def ensure_datetime(datetime_obj):
    """If possible/needed, return a real datetime."""
    try:
        return datetime_obj._to_real_datetime()
    except AttributeError:
        return datetime_obj


def multiline(s, strip_all_indents=False):
    if strip_all_indents:
        return " ".join([dedent(sub) for sub in s.strip().split("\n")])
    else:
        return dedent(s).strip().replace("\n", " ")


strip_multiline = partial(multiline, strip_all_indents=True)


def submit_array_job(filepath, ncpus, mem, walltime, max_index, show_only=False):
    """Submit an array job which runs the given file.

    The directory above is also added to the python path so that the 'specific' module
    that is assumed to be located there may be imported.

    Args:
        filepath (pathlib.Path): Path to the Python file to be executed as part of the
            array job.
        ncpus (int): Number of CPUs per job.
        mem (str): Memory per job.
        walltime (str): Walltime per job.
        max_index (int): Maximum array index (inclusive).
        show_only (bool): Print the job script instead of submitting it.

    """
    directory = filepath.parent
    job_name = filepath.with_suffix("").name
    output_dir = directory / Path(f"output_{job_name}")
    os.makedirs(output_dir, exist_ok=True)

    specific_dir = directory.parent
    assert list(
        specific_dir.glob("specific.py")
    ), "We expect to be 1 folder below 'specific.py'."

    job_script = f"""
#!/usr/bin/env bash

#PBS -N {job_name}
#PBS -l select=1:ncpus={ncpus}:mem={mem}
#PBS -l walltime={walltime}
#PBS -J 0-{max_index}
#PBS -e {output_dir}
#PBS -o {output_dir}

# Enable import of the right 'specific' module.
export PYTHONPATH={specific_dir}:$PYTHONPATH

# Finally, execute the script.
/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/python {filepath}
""".strip()
    if show_only:
        print(job_script)
        return

    with NamedTemporaryFile(prefix=f"{job_name}_", suffix=".sh") as job_file:
        with open(job_file.name, "w") as f:
            f.write(job_script)
        job_str = check_output(shlex.split(f"qsub -V {job_file.name}")).decode().strip()

    print(f"Submitted job {job_str}.")


def handle_array_job_args(filepath, func, **params):
    """Parse command line arguments as part of an array job.

    When submitting a task, `submit_array_job()` is invoked with the given filepath.
    Otherwise `func()` is called.

    Args:
        filepath (Path): Path to the Python file to be executed as part of the array
            job.
        func (callable): Callable with signature () that will be executed during the
            array job.
        **params: Parameters for `submit_array_job()`.

    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--submit", action="store_true", help="submit this array job")
    parser.add_argument(
        "--ncpus",
        type=int,
        help="how many cpus per job",
        default=params.get("ncpus", 1),
    )
    parser.add_argument(
        "--mem", help="memory per job, e.g. '5GB'", default=params.get("mem", "5GB")
    )
    parser.add_argument(
        "--walltime",
        help="walltime per job, e.g. '10:00:00'",
        default=params.get("walltime", "03:00:00"),
    )
    parser.add_argument(
        "--max-index",
        help="maximum job index (inclusive)",
        default=params.get("max_index", 100),
    )
    parser.add_argument(
        "--show-only", action="store_true", help="only show the job script"
    )
    args = parser.parse_args()
    if args.submit or args.show_only:
        submit_array_job(
            filepath,
            args.ncpus,
            args.mem,
            args.walltime,
            args.max_index,
            show_only=args.show_only,
        )
    else:
        func()


def shorten_features(features):
    """Abbreviate feature names.

    Args:
        features (str or iterable of str): Feature names to abbreviate.

    Returns:
        str: If `features` is of type `str`, the abbreviated string is returned.
        list of str: Otherwise, a list of abbreviated strings is returned.

    """
    if isinstance(features, str):
        return shorten_features((features,))[0]

    def month_repl(match_obj):
        return f"{match_obj.group(1)} M"

    def delta_repl(match_obj):
        return f"Δ{match_obj.group(1)} M"

    replacements = {
        "-(\d+) - .*Month$": delta_repl,
        "-(\d+) Month$": month_repl,
        "VOD Ku-band": "VOD",
        "Diurnal Temp Range": "DTR",
        "Dry Day Period": "Dry Days",
        re.escape("SWI(1)"): "SWI",
    }
    formatted = []
    for feature in features:
        for pattern, repl in replacements.items():
            feature = re.sub(pattern, repl, feature)
        formatted.append(feature)

    return formatted


def shorten_columns(df, inplace=False):
    """Apply `shorten_features()` to a DataFrame.

    Args:
        df (pandas DataFrame): DataFrame containing the columns to abbreviate.
        inplace (bool): Perform the rename operation inplace.

    Returns:
        pandas DataFrame: If `inplace` if False (default), the renamed DataFrame is
            returned.
        None: If `inplace` is True, None is returned.

    """
    return df.rename(
        columns=dict(
            (orig, short)
            for orig, short in zip(df.columns, shorten_features(df.columns))
        ),
        inplace=inplace,
    )


def replace_cube_coord(cube, new_coord, coord_name=None):
    """Name-based re-implementation of `iris.cube.Cube.replace_coord`.

    This relies on using `new_coord.name()` to retrieve the old coordinate (or
    `coord_name`, explicitly) instead of simply `new_coord` which fails to work for
    some cases.

    Args:
        cube (iris.cube.Cube): Cube for which to replace coordinates.
        new_coord (iris coord): New coordinate.
        coord_name (str, optional): Name of the coordinate to replace.

    Returns:
        iris.cube.Cube: The Cube containing the new coordinate is returned. Note that
        the operation is also performed in-place.

    """
    if coord_name is None:
        coord_name = new_coord.name()

    old_coord = cube.coord(coord_name)
    dims = cube.coord_dims(old_coord)
    was_dimensioned = old_coord in cube.dim_coords
    cube._remove_coord(old_coord)
    if was_dimensioned and isinstance(new_coord, iris.coords.DimCoord):
        cube.add_dim_coord(new_coord, dims[0])
    else:
        cube.add_aux_coord(new_coord, dims)

    for factory in cube.aux_factories:
        factory.update(old_coord, new_coord)

    return cube


def get_local_extrema(data, extrema_type="max"):
    """Determine the location of local extrema.

    Args:
        data (array-like): Data for which to find local extrema.
        extrema_type ({'max', 'min'}): If 'max', find local maxima. If 'min', find
            local minima.

    Returns:
        array-like: Boolean array that is True where a local minimum or maximum is
        located.

    Raises:
        ValueError: If `extrema_type` is not in {'max', 'min'}

    """
    if extrema_type == "max":
        # Find local maxima.
        op = np.less
    elif extrema_type == "min":
        op = np.greater
    else:
        raise ValueError(f"Unexpected value for extrema_type: {extrema_type}.")

    return op(np.diff(np.sign(np.diff(np.hstack((data[0], data, data[-1]))))), 0)


def get_local_maxima(data):
    """Return a boolean mask denoting the location of local maxima."""
    return get_local_extrema(data, "max")


def get_local_minima(data):
    """Return a boolean mask denoting the location of local minima."""
    return get_local_extrema(data, "min")


def significant_peak(x, diff_threshold=0.4, ptp_threshold=1):
    """Determine the existence of a 'significant' peak.

    This is determined using both the range of the given data and the characteristics
    of its local maxima.

    Args:
        x (array-like): Data to test.
        diff_threshold (float): Only applies if there are at least 2 local maxima. The
            heights of local maxima are calculated as the difference between the local
            maximum values and the lowest values of their surrounding minima. These
            heights are then divided by the height of the global maximum. If any other
            local maxima heights exceed `diff_threshold`, the global maximum is deemed
            not significant.
        ptp_threshold (float): If the range of `x` is lower than `ptp_threshold`, no
            peaks will be deemed significant.

    Returns:
        bool: True if there is a significant peak, False otherwise.

    """
    max_sample = np.max(x)
    min_sample = np.min(x)
    ptp = max_sample - min_sample

    if ptp < ptp_threshold:
        # If there is not enough variation, there is no significant peak.
        return False

    max_mask = get_local_maxima(x)

    if np.sum(max_mask) == 1:
        # If there is only one peak, there is nothing left to do.
        return True

    # If there are multiple peaks, we have to decide if these local maxima
    # are significant. If they are, there is no clearly defined
    # maximum for this sample.

    # Define significance of the minor peaks as the ratio between the
    # difference (peak value - local minima) and (peak value - local minima)
    # for the global maximum.

    min_indices = np.where(get_local_minima(x))[0]

    global_max_index = np.where(x == max_sample)[0][0]

    max_diffs = {}

    for max_index in np.where(max_mask)[0]:
        # Sample value at the local maximum.
        max_value = x[max_index]

        # Find the surrounding local minima.
        local_minima = []

        # Find preceding local minima, if any.
        if max_index > 0:
            local_minima.append(x[min_indices[min_indices < max_index][-1]])

        # Find following local minima, if any.
        if max_index < (len(x) - 1):
            local_minima.append(x[min_indices[min_indices > max_index][0]])

        max_diffs[max_index] = np.max(max_value - np.array(local_minima))

    global_max_diff = max_diffs[global_max_index]

    # Rescale using the maximum diff.
    for index, diff in max_diffs.items():
        max_diffs[index] = diff / global_max_diff

    if all(
        diff < diff_threshold
        for index, diff in max_diffs.items()
        if index != global_max_index
    ):
        return True
    else:
        return False
