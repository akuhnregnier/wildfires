# -*- coding: utf-8 -*-
import iris
import numpy as np
import pytest
from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime

from wildfires.data.datasets import (
    ERA5_DryDayPeriod,
    MonthlyDataset,
    UnexpectedCoordinateError,
    dummy_lat_lon_cube,
)

from .utils import data_availability


@pytest.mark.slow
@data_availability
def test_era_temporal_shifting():
    normal = ERA5_DryDayPeriod()
    shifted = normal.get_temporally_shifted_dataset(months=-3)

    # Since the shifted datasets represents previous months' data, there will not be
    # an overlap at the very beginning of the dataset, but rather at the end.

    # 2-month wide selection window.
    end_date = normal.max_time
    start_date = end_date - relativedelta(months=2)

    normal_cubes = normal.get_monthly_data(
        start=PartialDateTime(year=start_date.year, month=start_date.month),
        end=PartialDateTime(year=end_date.year, month=end_date.month),
    )

    # Shift window by 3 months to take into account the dataset shift.
    shift_end_date = end_date + relativedelta(months=3)
    shift_start_date = start_date + relativedelta(months=3)

    shifted_cubes = shifted.get_monthly_data(
        start=PartialDateTime(year=shift_start_date.year, month=shift_start_date.month),
        end=PartialDateTime(year=shift_end_date.year, month=shift_end_date.month),
    )

    for cube, shifted_cube in zip(normal_cubes, shifted_cubes):
        assert np.all(np.isclose(cube.data, shifted_cube.data))
        if any(hasattr(c, "mask") for c in (cube, shifted_cube)):
            assert all(hasattr(c.data, "mask") for c in (cube, shifted_cube))
            assert np.all(cube.data.mask == shifted_cube.data.mask)

        assert cube.attributes == shifted_cube.attributes


class DummyDataset(MonthlyDataset):
    def __init__(self):
        np.random.seed(1)
        self.cubes = iris.cube.CubeList(
            [dummy_lat_lon_cube(np.random.random((10, 10, 10)), monthly=True)]
        )


@pytest.mark.parametrize("deep", (False, True))
def test_temporal_shifting(deep):
    normal = DummyDataset()
    shifted = normal.get_temporally_shifted_dataset(months=-3, deep=deep)

    # Confirm that the underlying data is identical (not copied) if `deep=False`.
    if deep:
        assert normal.cube.data is not shifted.cube.data
    else:
        assert normal.cube.data is shifted.cube.data

    # Since the shifted datasets represents previous months' data, there will not be
    # an overlap at the very beginning of the dataset, but rather at the end.

    # 2-month wide selection window.
    end_date = normal.max_time
    start_date = end_date - relativedelta(months=2)

    normal_cubes = normal.get_monthly_data(
        start=PartialDateTime(year=start_date.year, month=start_date.month),
        end=PartialDateTime(year=end_date.year, month=end_date.month),
    )

    # Shift window by 3 months to take into account the dataset shift.
    shift_end_date = end_date + relativedelta(months=3)
    shift_start_date = start_date + relativedelta(months=3)

    shifted_cubes = shifted.get_monthly_data(
        start=PartialDateTime(year=shift_start_date.year, month=shift_start_date.month),
        end=PartialDateTime(year=shift_end_date.year, month=shift_end_date.month),
    )

    # Test that the data matches.
    for cube, shifted_cube in zip(normal_cubes, shifted_cubes):
        assert np.all(np.isclose(cube.data, shifted_cube.data))
        if any(hasattr(c, "mask") for c in (cube, shifted_cube)):
            assert all(hasattr(c.data, "mask") for c in (cube, shifted_cube))
            assert np.all(cube.data.mask == shifted_cube.data.mask)

        assert cube.attributes == shifted_cube.attributes


def test_other_temporal_coords():
    """No other temporal coordinates should be permitted prior to shifting.

    The shifting operation may or may not affect these auxiliary coordinates, and
    predicting how they may be affected is outside the scope of this function.

    """
    # Add a month_number coordinate to verify that an Exception should be raised.
    dataset = DummyDataset()
    iris.coord_categorisation.add_month_number(dataset.cube, "time")
    with pytest.raises(UnexpectedCoordinateError):
        # Attempt to get shifted dataset.
        dataset.get_temporally_shifted_dataset(months=-1)
