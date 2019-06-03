#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of code to be used throughout the project.

"""
import logging
import os
from time import time

import fiona
import numpy as np
from affine import Affine
from rasterio import features
from tqdm import tqdm

from wildfires.data.datasets import DATA_DIR

logger = logging.getLogger(__name__)


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


def land_mask(n_lon=1440):
    """Create land mask at the desired resolution.

    Data is taken from https://www.naturalearthdata.com/

    Args:
        n_lon (int): The number of longitude points of the final mask array. As the
            ratio between number of longitudes and latitudes has to be 2 in order for
            uniform scaling to work, the number of latitudes points is calculated as
            n_lon / 2.

    Returns:
        numpy.ndarray: Array of shape (n_lon / 2, n_lon) and dtype np.bool_. True
            where there is land, False otherwise.

    Examples:
        >>> import numpy as np
        >>> from wildfires.data.datasets import data_is_available
        >>> print(True)
        True
        >>> if data_is_available():
        ...     mask = land_mask(n_lon=1440)
        ...     assert mask.dtype == np.bool_
        ...     assert mask.sum() == 343928
        ...     assert mask.shape == (720, 1440)

    """
    assert n_lon % 2 == 0, (
        "The number of longitude points has to be an even number for the number of "
        "latitude points to be an integer."
    )
    n_lat = round(n_lon / 2)
    geom_np = np.zeros((n_lat, n_lon), dtype=np.uint8)
    with fiona.open(
        os.path.join(DATA_DIR, "land_mask", "ne_110m_land.shp"), "r"
    ) as shapefile:
        for geom in shapefile:
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
    Raises:
        ValueError: If `var` is an iterable of `single_type` and contains more than
            `elements` items.
        TypeError: If `var` is an iterable and its items are not all of type
            `single_type`.
        TypeError: If `fill_source` contains types other than int and NoneType.
        IndexError: If `len(fill_source)` < (`elements` - len(`var`)).
        IndexError: If `fill_source[-i]` is an int and
            `fill_source[-i]` >= `elements` - i.

    Returns:
        tuple: tuple with `elements` items.

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
                "Adding time dimension ({}) to broadcast land_mask.".format(
                    target_shape[0]
                )
            )
            new_mask = np.zeros(target_shape, dtype=np.bool_)
            new_mask += array.reshape(1, *array.shape)
            array = new_mask
        else:
            raise ValueError(
                "land_mask dimensions '{}' do not match cube dimensions '{}'".format(
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

    Raises:
        TypeError: If `strict` and `array` is of type `numpy.ndarray`. Regardless of
            `strict`, types other than `numpy.ma.core.MaskedArray` and `numpy.ndarray`
            will also raise a TypeError.

    Returns:
        numpy.ndarray: Flattened, unmasked data.

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
