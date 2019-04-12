#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cf_units
import iris
import numpy as np
import unittest

from wildfires.data.datasets import regrid
from wildfires.data.datasets import get_centres


class TestRegrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data.

        """
        start_data = np.array(
            [[3, 1, 1], [2, 2, 3], [1, 1, 1], [1, 3, 4]], dtype=np.float64
        )
        cls.source_cube = iris.cube.Cube(
            start_data,
            dim_coords_and_dims=(
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(90, -90, 5)),
                        standard_name="latitude",
                        units="degrees",
                    ),
                    0,
                ),
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(-180, 180, 4)),
                        standard_name="longitude",
                        units="degrees",
                    ),
                    1,
                ),
            ),
        )

        factor0 = np.sin((np.pi / 180) * 90) - np.sin((np.pi / 180) * 45)
        factor1 = np.sin((np.pi / 180) * 45) - np.sin((np.pi / 180) * 0)
        factor_sum = factor0 + factor1
        factor0 = factor0 / factor_sum
        factor1 = factor1 / factor_sum
        # Flip along after the equator due to symmetry.
        factor_arr = np.array([factor0, factor1, factor1, factor0]).reshape(4, 1)

        target_data = (cls.source_cube.data * factor_arr).reshape(2, 2, 3).sum(axis=1)

        cls.target_cube = iris.cube.Cube(
            target_data,
            dim_coords_and_dims=(
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(90, -90, 3)),
                        standard_name="latitude",
                        units="degrees",
                    ),
                    0,
                ),
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(-180, 180, 4)),
                        standard_name="longitude",
                        units="degrees",
                    ),
                    1,
                ),
            ),
        )

        # Set up cubes for time dimension testing.
        stacked_start_data = np.vstack([start_data[np.newaxis] for i in range(5)])
        stacked_target_data = np.vstack([target_data[np.newaxis] for i in range(5)])

        cls.time_source_cube = iris.cube.Cube(
            stacked_start_data,
            dim_coords_and_dims=(
                (
                    iris.coords.DimCoord(
                        np.arange(stacked_start_data.shape[0]),
                        standard_name="time",
                        units=cf_units.Unit("days since 1970-01-01 00:00:00"),
                    ),
                    0,
                ),
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(90, -90, 5)),
                        standard_name="latitude",
                        units="degrees",
                    ),
                    1,
                ),
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(-180, 180, 4)),
                        standard_name="longitude",
                        units="degrees",
                    ),
                    2,
                ),
            ),
        )

        cls.time_target_cube = iris.cube.Cube(
            stacked_target_data,
            dim_coords_and_dims=(
                (
                    iris.coords.DimCoord(
                        np.arange(stacked_target_data.shape[0]),
                        standard_name="time",
                        units=cf_units.Unit("days since 1970-01-01 00:00:00"),
                    ),
                    0,
                ),
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(90, -90, 3)),
                        standard_name="latitude",
                        units="degrees",
                    ),
                    1,
                ),
                (
                    iris.coords.DimCoord(
                        get_centres(np.linspace(-180, 180, 4)),
                        standard_name="longitude",
                        units="degrees",
                    ),
                    2,
                ),
            ),
        )

    def test_real_regrid(self):
        for method_bool in [True]:
            regridded = regrid(
                self.source_cube,
                area_weighted=method_bool,
                new_latitudes=self.target_cube.coord("latitude"),
                new_longitudes=self.target_cube.coord("longitude"),
            )

            self.assertTrue(np.all(np.isclose(regridded.data, self.target_cube.data)))

    def test_return_identical_cube(self):
        # NOTE: Does not apply for 3D data due to the current
        # implementation, but the 2D implementation is used recursively by
        # the 3D version.
        for method_bool in [True]:
            regridded = regrid(
                self.target_cube,
                area_weighted=method_bool,
                new_latitudes=self.target_cube.coord("latitude"),
                new_longitudes=self.target_cube.coord("longitude"),
            )

            self.assertEqual(
                id(regridded),
                id(self.target_cube),
                "(2D) The regridded and target cube should be identical.",
            )

    def test_time_dim_regrid(self):
        for method_bool in [True]:
            time_regridded = regrid(
                self.time_source_cube,
                area_weighted=method_bool,
                new_latitudes=self.time_target_cube.coord("latitude"),
                new_longitudes=self.time_target_cube.coord("longitude"),
            )

            self.assertTrue(
                np.all(np.isclose(time_regridded.data, self.time_target_cube.data))
            )


if __name__ == "__main__":
    unittest.main()
