# -*- coding: utf-8 -*-

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime

from wildfires.data.datasets import ERA5_DryDayPeriod
from wildfires.tests.test_datasets import data_availability


@pytest.mark.slow
@data_availability
def test_temporal_shifting():
    normal = ERA5_DryDayPeriod()
    shifted = ERA5_DryDayPeriod.get_temporally_shifted_dataset(months=-3)

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
