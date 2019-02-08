#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module that simplifies use of various datasets.

"""

from abc import ABC, abstractmethod
import os

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np


def get_centres(data):
    return (data[:-1] + data[1:]) / 2.


class Dataset(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_monthly_data(self):
        pass

    # NOTE: This is a good candidate for streamlining later on - most
    # implementations are going to be extremely similar, with the exception
    # being the time coordinate perhaps.
    @abstractmethod
    def regrid(self):
        pass


class AvitabileAGB(Dataset):

    def __init__(self):
        self.dir = os.path.join(
                os.path.expanduser('~'),
                'Data', 'Avitabile_AGB')
        # TODO: Initialise cube here (without loading data of course)?
        self.get_cube()

    def get_cube(self):
        self.cube = iris.load_cube(os.path.join(
            self.dir, 'Avitabile_AGB_Map_0d25.nc'))
        return self.cube

    def get_data(self):
        return self.get_cube().data

    def get_monthly_data(self, months=1):
        data = self.get_data()
        # TODO: Monthly padding/broadcasting depending on input argument.
        return data

    # TODO: Should be True by default
    def regrid(self, area_weighted=False):

        new_latitudes = get_centres(np.linspace(-90, 90, 60))
        new_longitudes = get_centres(np.linspace(0, 360, 100))

        if area_weighted:
            new_latitudes = iris.coords.DimCoord(
                    new_latitudes, standard_name='latitude',
                    units='degrees')
            new_longitudes = iris.coords.DimCoord(
                    new_longitudes, standard_name='longitude',
                    units='degrees')

            grid_coords = [
                (new_latitudes, 0),
                (new_longitudes, 1)
                ]

            new_grid = iris.cube.Cube(
                    np.zeros([coord[0].size for coord in grid_coords]),
                    dim_coords_and_dims=grid_coords)

            for coord in new_grid.coods():
                coord.guess_bounds()

            interpolated_cube = self.cube.regrid(
                    new_grid, iris.analysis.AreaWeighted())

        else:
            interp_points = [
                ('latitude', new_latitudes),
                ('longitude', new_longitudes)
                ]

            interpolated_cube = self.cube.interpolate(
                    interp_points, iris.analysis.Linear())

        return interpolated_cube


if __name__ == '__main__':
    # agb = AvitabileAGB()
