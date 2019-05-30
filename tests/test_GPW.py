#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime

from test_datasets import data_availability
from wildfires.data.datasets import GPW_v4_pop_dens


@data_availability
def test_gpw():
    dataset = GPW_v4_pop_dens()
    start_time = PartialDateTime(dataset.min_time.year, dataset.min_time.month)

    end_time = PartialDateTime(dataset.min_time.year + 1, dataset.min_time.month)

    monthly_cube = dataset.get_monthly_data(start_time, end_time)[0]

    interpolated_times = [
        monthly_cube.coord("time").cell(i).point for i in range(monthly_cube.shape[0])
    ]

    expected_times = [datetime(start_time.year, start_time.month, 1)]
    while expected_times[-1] != end_time:
        expected_times.append(expected_times[-1] + relativedelta(months=+1))

    assert interpolated_times == expected_times
