# -*- coding: utf-8 -*-
"""Collection of code to be used throughout the project.

"""
import logging
import os
from collections import Counter, namedtuple
from copy import deepcopy
from time import time

import fiona
import iris
import numpy as np
from affine import Affine
from rasterio import features
from scipy.ndimage import label
from tqdm import tqdm

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

    Examples:
        >>> in_360_longitude_system([0, 180, 360])
        True
        >>> in_360_longitude_system([0, 180])
        False
        >>> in_360_longitude_system([-180, 0, 180])
        False

    """
    longitudes = np.asarray(longitudes)
    if np.any(longitudes > 180):
        assert np.all(
            longitudes > -tol
        ), "If longitudes over 180 are present, there should be no longitudes below 0."
        return True
    assert np.all(
        longitudes > (-180 - tol)
    ), "Longitudes below -180 were found in the [-180, 180] system."
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
            `numpy.take_along_axis` for example.

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
            this iterable needs to match the number of indices.
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
    # To conform with the requirements of the `numpy.take_along_axis` function.
    indices_shape = [1] * len(cube.shape)
    indices_shape[axis] = -1

    indices = indices.reshape(*indices_shape)
    tr_new_data = np.take_along_axis(cube.data, indices, axis=axis)

    # Cut down to the required size, then substitute the reordered data.
    slices = [slice(None)] * len(cube.shape)
    slices[axis] = slice(indices.shape[axis])
    new_cube = deepcopy(cube)[tuple(slices)]
    new_cube.data = tr_new_data

    coord_name = [coord.name() for coord in cube.coords()][axis]
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
        axis (int, tuple of int, or None): Axes to subject to selection.
        longitudes (1-D iterable): Longitudes associated with the last axis of the
            data, ie. `len(longitudes) == data.shape[-1]`. If supplied, translation of
            the longitudes between systems ([-180, 180] <-> [0, 360]) will be
            considered to achieve the minimum possible resultant array size. If
            longitudes are supplied, both the (transformed - if needed) data and
            longitudes will be returned.

    Returns:
        (Translated) subset of `data` with same type [, (translated) longitudes].

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

    if axis is None:
        axis = tuple(range(len(data.shape)))
    elif isinstance(axis, (int, np.integer)):
        axis = (axis,)
    elif not isinstance(axis, tuple):
        raise ValueError(f"Invalid axis ('{axis}') type '{type(axis)}'.")

    slices = [slice(None)] * len(data.shape)

    all_axes = list(range(len(data.shape)))
    lon_ax = all_axes[-1]

    attempt_translation = False
    if longitudes is not None and lon_ax in axis:
        # Figure out if longitude translation is necessary separately.
        axis = tuple(x for x in axis if x != lon_ax)
        attempt_translation = True

    for ax in axis:
        # Compress mask such that only the axis of interest remains.
        compressed = np.all(mask, axis=tuple(x for x in all_axes if x != ax))

        elements, n_elements = label(compressed)
        if not n_elements:
            # If no masked elements were found there is nothing to do.
            slices[ax] = slice(None)
        else:
            ini_index = 0
            fin_index = data.shape[ax]

            # Check the beginning.
            if elements[0]:
                # Since False elements are labelled as 0, we know that the first element
                # is masked (True) here.

                # Count how many elements belong to this feature.
                ini_index += np.sum(elements == elements[0])

            if elements[-1]:
                fin_index -= np.sum(elements == elements[-1])

            slices[ax] = slice(ini_index, fin_index)

    indices = None

    if attempt_translation:
        # Check how many entries along the longitude axis could be removed for each of
        # the two cases.

        Result = namedtuple("Result", ("n_invalid", "ini_index", "fin_index"))

        # First check the native, untranslated case.

        # Compress mask such that only the axis of interest remains.
        compressed = np.all(mask, axis=tuple(x for x in all_axes if x != lon_ax))

        elements, n_elements = label(compressed)
        if not n_elements:
            # If no masked elements were found there is nothing to do.
            slices[lon_ax] = slice(None)
        else:
            ini_index = 0
            fin_index = data.shape[lon_ax]

            n_invalid = 0
            # Check the beginning.
            if elements[0]:
                # Since False elements are labelled as 0, we know that the first element
                # is masked (True) here.

                # Count how many elements belong to this feature.
                beginning_offset = np.sum(elements == elements[0])
                n_invalid += beginning_offset
                ini_index += beginning_offset

            if elements[-1]:
                end_offset = np.sum(elements == elements[-1])
                n_invalid += end_offset
                fin_index -= end_offset

            native_result = Result(n_invalid, ini_index, fin_index)

            # If there is a contiguous block of invalid values bigger than the
            # previous result, we have to carry out longitude translation to remove
            # it.
            if native_result.n_invalid >= max(
                Counter(
                    elements[native_result.ini_index : native_result.fin_index]
                ).values()
            ):
                slices[lon_ax] = slice(native_result.ini_index, native_result.fin_index)
            else:
                logger.info("Carrying out longitude translation.")

                # Carry out the translation.
                tr_longitudes, tr_indices = translate_longitude_system(
                    longitudes, return_indices=True
                )

                tr_indices = tr_indices.reshape(*[1] * (len(data.shape) - 1), -1)
                tr_mask = np.take_along_axis(mask, tr_indices, axis=lon_ax)

                # Compress mask such that only the axis of interest remains.
                compressed = np.all(
                    tr_mask, axis=tuple(x for x in all_axes if x != lon_ax)
                )

                elements, n_elements = label(compressed)
                assert n_elements, (
                    "We should only be here if there exists a bigger (by definition "
                    "with non-zero length) contiguous block of invalid elements after "
                    "translation."
                )

                ini_index = 0
                fin_index = data.shape[lon_ax]

                n_invalid = 0
                # Check the beginning.
                if elements[0]:
                    # Since False elements are labelled as 0, we know that the first element
                    # is masked (True) here.

                    # Count how many elements belong to this feature.
                    beginning_offset = np.sum(elements == elements[0])
                    n_invalid += beginning_offset
                    ini_index += beginning_offset

                if elements[-1]:
                    end_offset = np.sum(elements == elements[-1])
                    n_invalid += end_offset
                    fin_index -= end_offset

                translated_result = Result(n_invalid, ini_index, fin_index)

                # Compare the translated case with the native case.
                assert translated_result.n_invalid > native_result.n_invalid, (
                    "We should only be here if there exists a bigger contiguous block "
                    "of invalid elements after translation."
                )
                slices[lon_ax] = slice(
                    translated_result.ini_index, translated_result.fin_index
                )
                indices = tr_indices

    if longitudes is not None:
        if indices is None:
            return data[tuple(slices)], longitudes[slices[lon_ax]]
        else:
            # If a longitude translation was carried out, we need to translate the
            # data accordingly before we select it using the slices.

            # The `take_along_axis` operation is not compatible with iris Cubes.
            if isinstance(data, iris.cube.Cube):
                new_data = reorder_cube_coord(data, tr_indices, tr_longitudes, lon_ax)
            else:
                new_data = np.take_along_axis(data, tr_indices, axis=lon_ax)

            return new_data[tuple(slices)], tr_longitudes[slices[lon_ax]]
    else:
        return data[tuple(slices)]


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
