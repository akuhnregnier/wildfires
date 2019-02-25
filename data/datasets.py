#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module that simplifies use of various datasets.

"""

from abc import ABC, abstractmethod
import datetime
import os

import cf_units
import glob
import h5py
import iris
from iris.time import PartialDateTime
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = os.path.join(os.path.expanduser('~'), 'FIREDATA')


def get_centres(data):
    return (data[:-1] + data[1:]) / 2.


class Dataset(ABC):

    def __init__(self):
        self.dir = None
        self.cube = None

    def get_data(self):
        return self.cube.data

    def select_data(self, latitude_range=(-90, 90),
                    longitude_range=(-180, 180)):
        self.cube = (self.cube
                .intersection(latitude=latitude_range)
                .intersection(longitude=longitude_range))

    @abstractmethod
    def get_monthly_data(self):
        pass

    def regrid(
            self, area_weighted=True,
            new_latitudes=get_centres(np.linspace(-90, 90, 200)),
            new_longitudes=get_centres(np.linspace(-180, 180, 400))):

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
                    np.zeros([coord[0].points.size for coord in grid_coords]),
                    dim_coords_and_dims=grid_coords)

            for coord in new_grid.coords() + self.cube.coords():
                if not coord.has_bounds():
                    coord.guess_bounds()

            # import ipdb; ipdb.set_trace()
            interpolated_cube = self.cube.regrid(
                    new_grid, iris.analysis.AreaWeighted())

        else:
            interp_points = [
                ('latitude', new_latitudes),
                ('longitude', new_longitudes)
                ]

            interpolated_cube = self.cube.interpolate(
                    interp_points, iris.analysis.Linear())

        self. cube = interpolated_cube


class AvitabileAGB(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'Avitabile_AGB')
        self.cube = iris.load_cube(os.path.join(
            self.dir, 'Avitabile_AGB_Map_0d25.nc'))
        self.select_data(
                # latitude_range=(0, 40)
                )

    def get_monthly_data(self, months=1):
        data = self.get_data()
        # TODO: Monthly padding/broadcasting depending on input argument.
        pass


class AvitabileThurnerAGB(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'AvitabileThurner-merged_AGB')
        self.cube = iris.load_cube(os.path.join(
            self.dir, 'Avi2015-Thu2014-merged_AGBtree.nc'))
        self.select_data(
                # latitude_range=(0, 40)
                )

    def get_monthly_data(self, months=1):
        raise NotImplementedError()


class CarvalhaisGPP(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'Carvalhais_VegC-TotalC-Tau')
        self.cube = iris.load_cube(os.path.join(
            self.dir, 'Carvalhais.gpp_50.360.720.1.nc'))

    def get_monthly_data(self):
        raise NotImplementedError("Only mean data available!")


class ESA_CCI_Fire(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-Fire_burnedarea')
        self.cube = iris.load_cube(os.path.join(
                self.dir, 'MODIS_cci.BA.2001.2016.1440.720.365days.sum.nc'))

    def get_monthly_data(self):
        raise NotImplementedError("Only yearly data available!")


class ESA_CCI_Landcover(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-LC_landcover',
                                '0d25_landcover')
        self.cube = iris.load_cube(os.path.join(
                self.dir, 'MODIS_cci.BA.2001.2016.1440.720.365days.sum.nc'))

    def get_monthly_data(self):
        raise NotImplementedError("Only yearly data available!")


class ESA_CCI_Landcover_PFT(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-Fire_burnedarea')
        self.cube = iris.load_cube(os.path.join(
                self.dir, 'MODIS_cci.BA.2001.2016.1440.720.365days.sum.nc'))

    def get_monthly_data(self):
        raise NotImplementedError("Only yearly data available!")


class ESA_CCI_Soilmoisture(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-Fire_burnedarea')
        self.cube = iris.load_cube(os.path.join(
                self.dir, 'MODIS_cci.BA.2001.2016.1440.720.365days.sum.nc'))

    def get_monthly_data(self):
        raise NotImplementedError("Only yearly data available!")




class GFEDv4s(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'gfedv4', 'data')

        filenames = glob.glob(os.path.join(self.dir, '*.hdf5'))
        filenames.sort()  # increasing years

        # for each file (each year), load the data, the latitudes and
        # longitudes and place them into a cube
        years = []
        data = []
        for f in filenames:
            year = int(f[-9:-5])
            years.append(year)
            container = h5py.File(f)

            for month_str in [format(m, '02d') for m in range(1, 13)]:
                data.append(container['burned_area'][month_str]
                            ['burned_fraction'][()][None, ...])

        assert years == sorted(years), 'Should be monotonically increasing'

        # use the last file (of previous for loop) to get latitudes and
        # longitudes, assuming that they are the same for all the data
        # files!
        latitudes = container['lat'][()]
        longitudes = container['lon'][()]

        # make sure that the lats and lons are uniform along the grid
        assert np.all(longitudes[0] == longitudes)
        assert np.all(latitudes.T[0] == latitudes.T)

        longitudes = iris.coords.DimCoord(
                longitudes[0], standard_name='longitude',
                units='degrees')
        latitudes = iris.coords.DimCoord(
                latitudes.T[0], standard_name='latitude',
                units='degrees')

        time_unit_str = 'hours since 1970-01-01 00:00:00'
        calendar = 'gregorian'
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        num_times = []
        for m in range(len(data)):
            month = (m % 12) + 1
            year = (m // 12) + min(years)
            assert year <= max(years)
            num_times.append(cf_units.date2num(
                datetime.datetime(year, month, 1),
                time_unit_str, calendar))

        times = iris.coords.DimCoord(
                num_times, standard_name='time',
                units=time_unit)

        for coord in (longitudes, latitudes, times):
            coord.guess_bounds()

        self.cube = iris.cube.Cube(
                np.vstack(data), dim_coords_and_dims=[
                    (times, 0),
                    (latitudes, 1),
                    (longitudes, 2)
                    ],
                # standard_name='burned area fraction',
                # units='fraction'
                )

        self.select_data(
                # latitude_range=(0, 40)
                )

    def get_monthly_data(self, months=1):
        return self.cube.data


if __name__ == '__main__':
    # agb = AvitabileThurnerAGB()
    # plt.close('all')
    # plt.figure()
    # qplt.contourf(agb.cube, 20)
    # plt.gca().coastlines()
    # new = agb.regrid()
    # plt.figure()
    # qplt.contourf(new, 20)
    # plt.gca().coastlines()
    # plt.show()

    # fire = GFEDv4s()

    # fire.regrid()
    # agb.regrid()

    # agb_data = np.zeros_like(fire.get_data()) + agb.get_data()[None, ...]

    # plt.close('all')
    # plt.figure()
    # qplt.contourf(CarvalhaisGPP().cube[0], 20)
    # plt.gca().coastlines()
    # plt.title('Carvalhais GPP')

    # plt.figure()
    # qplt.contourf(ESA_CCI_Fire().cube[0], 20)
    # plt.gca().coastlines()
    # plt.title('ESA Fire')


