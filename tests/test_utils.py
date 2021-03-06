# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose

from wildfires.data.datasets import dummy_lat_lon_cube
from wildfires.utils import (
    get_batches,
    get_centres,
    get_local_extrema,
    get_local_maxima,
    get_local_minima,
    select_valid_subset,
    shorten_features,
    significant_peak,
    translate_longitude_system,
)

from .utils import allequal


def test_subset():
    np.random.seed(1)
    data = np.random.random((100, 100))
    data = np.ma.MaskedArray(data, mask=np.zeros_like(data, dtype=np.bool_))
    data.mask[:40] = True
    data.mask[:, :20] = True
    data.mask[:, -10:] = True

    assert_allclose(select_valid_subset(data), data[40:, 20:90])
    assert_allclose(select_valid_subset(data, axis=0), data[40:])
    assert_allclose(select_valid_subset(data, axis=1), data[:, 20:90])
    assert_allclose(select_valid_subset(data, axis=(0, 1)), data[40:, 20:90])


def test_longitude_system():
    # -180, 180 case.
    data = np.array([1, 2, 3, 4, 5])
    longitudes = np.array([-180, -90, 0, 90, 170])
    new_longitudes, indices = translate_longitude_system(
        longitudes, return_indices=True
    )
    new_data = data[indices]
    assert_allclose(new_longitudes, [0, 90, 170, 180, 270])
    assert_allclose(new_data, [3, 4, 5, 1, 2])

    # 0, 360 case
    data = np.array([3, 4, 5, 1, 2])
    longitudes = np.array([0, 90, 170, 180, 270])
    new_longitudes, indices = translate_longitude_system(
        longitudes, return_indices=True
    )
    new_data = data[indices]
    assert_allclose(new_longitudes, [-180, -90, 0, 90, 170])
    assert_allclose(new_data, [1, 2, 3, 4, 5])


def test_cube_translation_subset():
    np.random.seed(1)
    data = np.random.random((1, 10, 10))
    data = np.ma.MaskedArray(data, mask=np.zeros_like(data, dtype=np.bool_))
    # Invalidate elements at the beginning of the latitude range.
    data.mask[:, :2] = True
    # Invalidate elements in the middle of the longitude range.
    data.mask[..., 3:6] = True

    cube = dummy_lat_lon_cube(data)

    sub_cube, _ = select_valid_subset(cube, longitudes=cube.coord("longitude").points)

    # Piece data together again in the expected manner.
    comp_data = np.ma.concatenate((data[:, 2:, 6:], data[:, 2:, :3]), axis=-1)

    # Compare data.
    assert_allclose(comp_data, sub_cube.data)
    # Compare data mask.
    assert not comp_data.mask
    assert not np.any(sub_cube.data.mask)

    # Compare ordering of latitudes.
    assert_allclose(
        cube.coord("latitude")[2:].points, sub_cube.coord("latitude").points
    )
    assert_allclose(
        cube.coord("latitude")[2:].bounds, sub_cube.coord("latitude").bounds
    )

    # Compare ordering of longitudes. This requires manual translation.
    old_lons = cube.coord("longitude").points
    assert_allclose(
        list(old_lons[6:]) + list(old_lons[:3] + 360),
        sub_cube.coord("longitude").points,
    )

    # Check longitude bounds.
    assert_allclose(
        sub_cube.coord("longitude").bounds,
        np.array(
            [
                [36.0, 72.0],
                [72.0, 108.0],
                [108.0, 144.0],
                [144.0, 180.0],
                [180.0, 216.0],
                [216.0, 252.0],
                [252.0, 288.0],
            ]
        ),
    )


def test_data_translation_subset():
    np.random.seed(1)
    data = np.random.random((1, 10, 10))
    data = np.ma.MaskedArray(data, mask=np.zeros_like(data, dtype=np.bool_))
    # Mark elements at the beginning of the latitude range, but in the middle of the
    # longitude range as invalid.
    data.mask[:, :2] = True
    data.mask[..., 3:6] = True

    sub_data, _ = select_valid_subset(
        data, longitudes=get_centres(np.linspace(-180, 180, data.shape[-1] + 1))
    )

    comp_data = data[:, 2:]
    comp_data = np.ma.concatenate((comp_data[..., 6:], comp_data[..., :3]), axis=-1)

    # Compare data.
    assert_allclose(comp_data, sub_data.data)
    # Compare data mask.
    assert not comp_data.mask
    assert not np.any(sub_data.mask)


def test_shorten_features():
    assert shorten_features("Diurnal Temp Range") == "DTR"
    assert shorten_features(["Diurnal Temp Range"]) == ["DTR"]
    assert shorten_features(["Diurnal Temp Range", "Dry Day Period"]) == [
        "DTR",
        "DD",
    ]

    assert shorten_features("VOD Ku-band") == "VOD"
    assert shorten_features("SWI(1)") == "SWI"

    assert shorten_features("SWI(1) -1 Month") == "SWI 1M"
    assert shorten_features("SWI(1) -18 - -6 Month") == "SWI Δ18M"


def test_local_maxima():
    assert allequal(
        get_local_maxima([0, 1, 2, 1, 3]), np.array([0, 0, 1, 0, 1], dtype=bool)
    )


def test_local_minima():
    assert allequal(
        get_local_minima([0, 1, 2, 1, 3]), np.array([1, 0, 0, 1, 0], dtype=bool)
    )


def test_local_extrema():
    assert allequal(get_local_extrema([0, 1, 2, 1, 2]), np.array([1, 0, 1, 1, 1]))
    assert allequal(get_local_extrema([0, 1, 2, -1, 2]), np.array([1, 0, 1, 1, 1]))


def test_significant_peak():
    assert not significant_peak([0, 1], ptp_threshold=10)
    assert significant_peak([0, 1], ptp_threshold=0.1)
    assert not significant_peak(
        [0, 1, 0, 0.5, 0.3], diff_threshold=0.4, ptp_threshold=0.5
    )
    assert significant_peak([0, 1, 0, 0.5, 0.3], diff_threshold=0.6, ptp_threshold=0.5)
    assert allequal(
        np.sort(significant_peak([1, 0.4, -0.5, 0.1, 0.3], 0.5, 0, False)),
        np.array([0, 2]),
    )
    assert allequal(
        np.sort(significant_peak([1, 0.4, -0.5, 0.7, 0.3], 0.5, 0, False)),
        np.array([0, 2, 3]),
    )
    assert allequal(
        np.sort(significant_peak([1, 0.4, -0.5, 0.7, 0.3], 0.6, 0, False)),
        np.array([0, 3]),
    )


def test_get_batches():
    batches = list(get_batches(list(range(10)), 3))
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8, 9]
