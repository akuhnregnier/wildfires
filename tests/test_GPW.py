#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime
import unittest

from wildfires.data.datasets import GPW_v4_pop_dens


class TestGWP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = GPW_v4_pop_dens()
        cls.start_time = PartialDateTime(
                cls.dataset.min_time.year,
                cls.dataset.min_time.month)

        cls.end_time = PartialDateTime(
                cls.dataset.min_time.year + 1,
                cls.dataset.min_time.month)

        cls.monthly_cube = cls.dataset.get_monthly_data(
                cls.start_time, cls.end_time)

        cls.interpolated_times = [
                cls.monthly_cube.coord('time').cell(i).point for i in
                range(cls.monthly_cube.shape[0])]

        cls.expected_times = [
                datetime(cls.start_time.year, cls.start_time.month, 1)]
        while cls.expected_times[-1] != cls.end_time:
            cls.expected_times.append(
                    cls.expected_times[-1] + relativedelta(months=+1))

    def test_monthly_data(self):
        self.assertEqual(
                self.interpolated_times,
                self.expected_times)


if __name__ == '__main__':
    unittest.main()
