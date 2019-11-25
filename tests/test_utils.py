# -*- coding: utf-8 -*-
import numpy as np

from wildfires.data.datasets import dummy_lat_lon_cube, get_centres
from wildfires.utils import select_valid_subset, translate_longitude_system


def test_subset():
    np.random.seed(1)
    data = np.random.random((100, 100))
    data = np.ma.MaskedArray(data, mask=np.zeros_like(data, dtype=np.bool_))
    data.mask[:40] = True
    data.mask[:, :20] = True
    data.mask[:, -10:] = True

    assert np.all(np.isclose(select_valid_subset(data), data[40:, 20:90]))
    assert np.all(np.isclose(select_valid_subset(data, axis=0), data[40:]))
    assert np.all(np.isclose(select_valid_subset(data, axis=1), data[:, 20:90]))
    assert np.all(np.isclose(select_valid_subset(data, axis=(0, 1)), data[40:, 20:90]))


def test_longitude_system():
    # -180, 180 case.
    data = np.array([1, 2, 3, 4, 5])
    longitudes = np.array([-180, -90, 0, 90, 170])
    new_longitudes, indices = translate_longitude_system(
        longitudes, orig="-180", return_indices=True
    )
    new_data = data[indices]
    assert np.all(np.isclose(new_longitudes, [0, 90, 170, 180, 270]))
    assert np.all(np.isclose(new_data, [3, 4, 5, 1, 2]))

    # 0, 360 case
    data = np.array([3, 4, 5, 1, 2])
    longitudes = np.array([0, 90, 170, 180, 270])
    new_longitudes, indices = translate_longitude_system(
        longitudes, orig="0", return_indices=True
    )
    new_data = data[indices]
    assert np.all(np.isclose(new_longitudes, [-180, -90, 0, 90, 170]))
    assert np.all(np.isclose(new_data, [1, 2, 3, 4, 5]))


def test_cube_translation_subset():
    np.random.seed(1)
    data = np.random.random((1, 10, 10))
    data = np.ma.MaskedArray(data, mask=np.zeros_like(data, dtype=np.bool_))
    # Mark elements at the beginning of the latitude range, but in the middle of the
    # longitude range as invalid.
    data.mask[:, :2] = True
    data.mask[..., 3:6] = True

    cube = dummy_lat_lon_cube(data)

    sub_cube, _ = select_valid_subset(cube, longitudes=cube.coord("longitude").points)

    comp_data = data[:, 2:]
    comp_data = np.ma.concatenate((comp_data[..., 6:], comp_data[..., :3]), axis=-1)

    # Compare data.
    assert np.all(np.isclose(comp_data, sub_cube.data))
    # Compare data mask.
    assert not comp_data.mask
    assert not np.any(sub_cube.data.mask)

    # Compare ordering of latitudes.
    assert np.all(
        np.isclose(cube.coord("latitude")[2:].points, sub_cube.coord("latitude").points)
    )
    assert np.all(
        np.isclose(cube.coord("latitude")[2:].bounds, sub_cube.coord("latitude").bounds)
    )

    # Compare ordering of longitudes. This requires manual translation.
    old_lons = cube.coord("longitude").points
    assert np.all(
        np.isclose(
            list(old_lons[6:]) + list(old_lons[:3] + 360),
            sub_cube.coord("longitude").points,
        )
    )

    # Check longitude bounds.
    assert np.all(
        np.isclose(
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
    assert np.all(np.isclose(comp_data, sub_data.data))
    # Compare data mask.
    assert not comp_data.mask
    assert not np.any(sub_data.mask)