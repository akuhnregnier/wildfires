#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import iris
from wildfires.data.datasets import regrid
from wildfires.data.datasets import get_centres

class TestRegrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test data.

        """
        cls.source_cube = iris.cube.Cube(
            np.array([
                [3, 1, 1],
                [2, 2, 3],
                [1, 1, 1],
                [1, 3, 4]], dtype=np.float64),
            dim_coords_and_dims=(
                (iris.coords.DimCoord(
                    get_centres(np.linspace(90, -90, 5)),
                    standard_name='latitude',
                    units='degrees'), 0),
                (iris.coords.DimCoord(
                    get_centres(np.linspace(-180, 180, 4)),
                    standard_name='longitude',
                    units='degrees'), 1),
                ))

        factor0 = np.sin((np.pi/180) * 90) - np.sin((np.pi/180) * 45)
        factor1 = np.sin((np.pi/180) * 45) - np.sin((np.pi/180) * 0)
        factor_sum = factor0 + factor1
        factor0 = factor0 / factor_sum
        factor1 = factor1 / factor_sum
        # Flip along after the equator due to symmetry.
        factor_arr = np.array(
                [factor0, factor1, factor1, factor0]).reshape(4, 1)

        target_data = (cls.source_cube.data * factor_arr
                ).reshape(2, 2, 3).sum(axis=1)

        cls.target_cube = iris.cube.Cube(
            target_data,
            dim_coords_and_dims=(
                (iris.coords.DimCoord(
                    get_centres(np.linspace(90, -90, 3)),
                    standard_name='latitude',
                    units='degrees'), 0),
                (iris.coords.DimCoord(
                    get_centres(np.linspace(-180, 180, 4)),
                    standard_name='longitude',
                    units='degrees'), 1),
                ))

    def test_real_regrid(self):
        regridded = regrid(
                self.source_cube, area_weighted=True,
                new_latitudes=self.target_cube.coord('latitude'),
                new_longitudes=self.target_cube.coord('longitude'))

        self.assertTrue(np.all(np.isclose(
            regridded.data,
            self.target_cube.data)))

    def test_return_identical_cube(self):
        regridded = regrid(
                self.target_cube, area_weighted=True,
                new_latitudes=self.target_cube.coord('latitude'),
                new_longitudes=self.target_cube.coord('longitude'))

        self.assertEqual(
                id(regridded), id(self.target_cube),
                "The regridded and target cube should be the same cube.")


if __name__ == '__main__':
    unittest.main()
