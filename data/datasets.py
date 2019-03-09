#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module that simplifies use of various datasets.

"""

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob
import logging
import operator
import os
import pickle
import re
import warnings

import cf_units
from git import Repo
import h5py
import iris
import iris.coord_categorisation
from iris.time import PartialDateTime
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
import rasterio
import statsmodels.api as sm
from tqdm import tqdm
# import statsmodels.genmod.families.links as links


DATA_DIR = os.path.join(os.path.expanduser('~'), 'FIREDATA')
pickle_file = os.path.join(DATA_DIR, 'cubes.pickle')

repo_dir = os.path.join(os.path.dirname(__file__), os.pardir)
repo = Repo(repo_dir)


def dummy_lat_lon_cube(data, lat_lims=(-90, 90), lon_lims=(-180, 180),
                       **kwargs):
    assert len(data.shape) == 2
    new_latitudes = get_centres(np.linspace(*lat_lims, data.shape[0] + 1))
    new_longitudes = get_centres(np.linspace(*lon_lims, data.shape[1] + 1))
    new_lat_coord = iris.coords.DimCoord(
            new_latitudes, standard_name='latitude',
            units='degrees')
    new_lon_coord = iris.coords.DimCoord(
            new_longitudes, standard_name='longitude',
            units='degrees')

    grid_coords = [
        (new_lat_coord, 0),
        (new_lon_coord, 1)
        ]
    return iris.cube.Cube(data, dim_coords_and_dims=grid_coords, **kwargs)


def data_map_plot(data, lat_lims=(-90, 90), lon_lims=(-180, 180),
                  **kwargs):
    """Used to plot data or an iris.cube.Cube on a map with coastlines.

    """
    if isinstance(data, iris.cube.Cube):
        cube = data
    else:
        cube = dummy_lat_lon_cube(data, lat_lims, lon_lims, **kwargs)

    fig = plt.figure()
    qplt.contourf(cube)
    plt.gca().coastlines()
    return fig


def load_cubes(files, n=None):
    """Similar to iris.load(), but seems to scale much better with the
    number of files to load.

    """
    # TODO: Avoid double sorting
    # NOTE: The order in which iris.load() loads cubes into a CubeList is
    # not constant but varies from one execution to the next (presumably
    # due to a random seed of some sort)!

    # Make sure files are sorted so that times increase.
    files.sort()
    l = iris.cube.CubeList()
    for f in tqdm(files[slice(0, n, 1)]):
        l.extend(iris.load(f))
    return l


def get_centres(data):
    """Get the elements between elements of an array.

    Examples:
        >>> from wildfires.data.datasets import get_centres
        >>> import numpy as np
        >>> a = np.array([1,2,3])
        >>> b = get_centres(a)
        >>> np.all(np.isclose(b, np.array([1.5, 2.5])))
        True

    """
    return (data[:-1] + data[1:]) / 2.


def regrid(
        cube, area_weighted=True,
        new_latitudes=get_centres(np.linspace(-90, 90, 721)),
        new_longitudes=get_centres(np.linspace(-180, 180, 1441))):
    """Keep (optional) time coordinate, but regrid latitudes and longitudes.

    Expects either (time, lat, lon) or (lat, lon) coordinates.

    NOTE: Regarding caching AreaWeighted regridders - the creation of the
    regridder does not seem to take much time, however, so this step is
    almost inconsequential. Furthermore, as time coordinate differences may
    exist between coordinates, iris does not support such differences with
    cached regridders.

    """
    n_dim = len(cube.shape)
    assert n_dim in {2, 3}, "Need [[time,] lat, lon] dimensions."

    for coord in [c for c in cube.coords() if c.name() in
                  ['time', 'latitude', 'longitude'][3 - n_dim:]]:
        if not coord.has_bounds():
            coord.guess_bounds()

    # Using getattr allows the input coordinates to be both
    # iris.coords.DimCoord (with a 'points' attribute) as well as normal
    # numpy arrays.
    new_latitudes = iris.coords.DimCoord(
            getattr(new_latitudes, 'points', new_latitudes),
            standard_name='latitude', units='degrees')
    new_longitudes = iris.coords.DimCoord(
            getattr(new_longitudes, 'points', new_longitudes),
            standard_name='longitude', units='degrees')

    # If there is a time dimension, n_dim = 3, and so there will be 3
    # entries. However, if there is no time dimension, n_dim = 2, and the
    # first entry will be clipped.
    interp_points = [
        ('time', cube.coords()[0]),
        ('latitude', new_latitudes),
        ('longitude', new_longitudes)
        ][3 - n_dim:]

    orig_coord_dict = dict(
            [(name, tuple(cube.coord(name).points)) for name in
             ['latitude', 'longitude']])

    new_coord_dict = dict(
            [(name, tuple(dict(interp_points)[name].points)) for name in
             ['latitude', 'longitude']])

    if orig_coord_dict == new_coord_dict:
        logging.info("Identical input output, not regridding.")
        return cube

    if area_weighted:

        # If n_dim = 2, lats and lons are at positions 0 and 1, not 1 and 2
        # (using the modulo operator).
        grid_coords = [
            (cube.coords()[0], 0),
            (new_latitudes, 1 - (3 % n_dim)),
            (new_longitudes, 2 - (3 % n_dim))
            ][3 - n_dim:]

        new_grid = iris.cube.Cube(
                np.zeros([coord[0].points.size for coord in grid_coords]),
                dim_coords_and_dims=grid_coords)

        for coord in new_grid.coords():
            if not coord.has_bounds():
                coord.guess_bounds()

        interpolated_cube = cube.regrid(
                new_grid, iris.analysis.AreaWeighted())

    else:
        interpolated_cube = cube.interpolate(
                interp_points, iris.analysis.Linear())

    return interpolated_cube


def combine_masks(data, invalid_values=(0,)):
    """Create a mask that shows where data is invalid.

    NOTE: Calls data.data, so lazy data is realised here!

    True - invalid data
    False - valid data

    Returns a boolean array with the same shape as the input data.

    """
    if hasattr(data, 'mask'):
        mask = data.mask
        data_arr = data.data
    else:
        mask = np.zeros_like(data, dtype=bool)
        data_arr = data

    for invalid_value in invalid_values:
        mask |= (data_arr == invalid_value)

    return mask


def monthly_constraint(
        t, time_range=(PartialDateTime(2000, 1), PartialDateTime(2010, 1)),
        inclusive_lower=True, inclusive_upper=True):
    """Constraint function which ignores the day and only considers the
    year and month.

    """
    lower_op = operator.ge if inclusive_lower else operator.gt
    upper_op = operator.le if inclusive_upper else operator.lt
    comp_datetime = PartialDateTime(year=t.year, month=t.month)

    return (lower_op(comp_datetime, time_range[0]) and
            upper_op(comp_datetime, time_range[1]))


class Dataset(ABC):

    def __init__(self):
        # self.dir = None
        # self.cubes = None
        raise NotImplementedError()

    @property
    def min_time(self):
        return self.cubes[0].coord('time').cell(0).point

    @property
    def max_time(self):
        return self.cubes[0].coord('time').cell(-1).point

    def get_data(self):
        """Returns either lazy data (dask array) or a numpy array.

        """
        return self.cube.core_data()

    @property
    def cache_filename(self):
        return os.path.join(DATA_DIR, 'cache', type(self).__name__ + '.nc')

    @staticmethod
    def save_data(cache_data, target_filename):
        """Save as NetCDF file.

        Args:
            cache_data (iris.cube.Cube or iris.cube.CubeList): This will be
                saved as a NetCDF file.
            target_filename (str): The filename that the data will be saved
                to. Must end in '.nc', since the data is meant to be saved
                as a NetCDF file.

        Returns:
            str or None: The current hex commit sha hash of the repo if a
            new file was created. Otherwise, if the file was already there
            and not overwritten, None is returned.

        """
        assert target_filename[-3:] == '.nc', (
                "Data must be saved as a NetCDF file, got:'{:}'"
                .format(target_filename))
        assert isinstance(cache_data, (iris.cube.Cube, iris.cube.CubeList)), (
                "Data to be saved must either be a Cube or a CubeList. "
                "Got:{:}".format(cache_data))

        if isinstance(cache_data, iris.cube.Cube):
            cache_data = iris.cube.CubeList([cache_data])

        if os.path.isfile(target_filename):
            # TODO: Want to overwrite if the commit hash is different?
            # Maybe add a flag to do this.
            logging.info("File exists, not overwriting:'{:}'"
                         .format(target_filename))
        else:
            assert (not repo.untracked_files) and (not repo.is_dirty()), (
                    "All changes must be committed and all files must be "
                    "tracked.")

            # Note down the commit sha hash so that the code used to
            # generate the cached data can be retrieved easily later on.
            for cube in cache_data:
                cube.attributes['commit'] = repo.head.ref.commit.hexsha

            if not os.path.isdir(os.path.dirname(target_filename)):
                os.makedirs(os.path.dirname(target_filename))
            logging.info("Saving cubes to:'{:}'".format(target_filename))
            iris.save(cache_data, target_filename, zlib=True)
            return cube.attributes['commit']

    @staticmethod
    def read_data(target_filename):
        """Read from NetCDF file.

        Args:
            target_filename (str): The filename that the data will be saved
                to. Must end in '.nc', since the data is meant to be saved
                as a NetCDF file.

        Raises:
            AssertionError: If the commit hashes of the cubes that are
                loaded do not match.

        """
        if os.path.isfile(target_filename):
            cubes = iris.load(target_filename)
            if not cubes:
                os.remove(target_filename)
                logging.warning("No cubes were found. Deleted file:{:}"
                                .format(target_filename))
                return

            commit_hashes = [cube.attributes['commit'] for cube in cubes]
            assert len(set(commit_hashes)) == 1, (
                    "Cubes should all stem from the same commit.")
            logging.debug(
                    "File exists, returning cubes from:'{:}'"
                    .format(target_filename))
            return cubes
        else:
            logging.info("File does not exist:'{:}'"
                         .format(target_filename))

    def write_cache(self):
        """Write list of cubes to disk as a NetCDF file using iris.

        Also record the git commit id that the data was generated with,
        making sure that there are no uncommitted changes in the repository
        at the time.

        """
        self.save_data(self.cubes, self.cache_filename)

    def read_cache(self):
        cubes = self.read_data(self.cache_filename)
        if cubes:
            self.cubes = cubes
            logging.info(
                    "File exists, returning cubes from:'{:}' -> Dataset "
                    "timespan {:} -- {:}. Generated using commit {:}"
                    .format(self.cache_filename, self.min_time,
                            self.max_time, self.cubes[0].attributes['commit']))
            return self.cubes

    def select_data(self, latitude_range=(-90, 90),
                    longitude_range=(-180, 180)):
        self.cube = (self.cube
                     .intersection(latitude=latitude_range)
                     .intersection(longitude=longitude_range))
        return self.cube

    @abstractmethod
    def get_monthly_data(self):
        pass


class AvitabileAGB(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'Avitabile_AGB')
        self.cube = iris.load_cube(os.path.join(
            self.dir, 'Avitabile_AGB_Map_0d25.nc'))

    def get_monthly_data(self):
        raise NotImplementedError("Data is static.")


class AvitabileThurnerAGB(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'AvitabileThurner-merged_AGB')
        self.cube = iris.cube.CubeList(iris.load_cube(os.path.join(
            self.dir, 'Avi2015-Thu2014-merged_AGBtree.nc')))

    def get_monthly_data(self):
        raise NotImplementedError("Data is static.")


class CarvalhaisGPP(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'Carvalhais_VegC-TotalC-Tau')
        self.cubes = iris.cube.CubeList([iris.load_cube(
            os.path.join(self.dir, 'Carvalhais.gpp_50.360.720.1.nc'))])

    def get_monthly_data(self):
        raise NotImplementedError("Data is static.")


class CHELSA(Dataset):
    """For primary analysis, it is advisable to use hpc
    (cx1_scipts/run_chelsa_script.sh) in order to process the tif files
    into nc files as a series of jobs, which would take an incredibly long
    time otherwise (on the order of days).

    Once that script has been run, the resulting nc files can be used to
    easily construct a large iris Cube containing all the data.

    """

    def __init__(self, process_slice=slice(None)):
        """Initialise the cubes.

        Args:
            process_slice (slice): Used to limit the loading/processing of
                raw data .tif data files. Slices resulting in single
                elements (eg. slice(i, i+1)) can be provided with i being
                the PBS array job index (for example) to quickly generate
                all the required .nc files from the .tif files using array
                jobs on the hpc.

        """
        self.dir = os.path.join(DATA_DIR, 'CHELSA')

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        files = glob.glob(os.path.join(self.dir, '**', '*.tif'),
                          recursive=True)
        files.sort()

        mapping = {
                'prec': {
                    'scale': 1,
                    'unit': cf_units.Unit('mm/month'),
                    'long_name': 'monthly precipitation',
                    },
                'tmax': {
                    'scale': 0.1,
                    'unit': cf_units.Unit('degrees Celsius'),
                    'long_name': 'maximum temperature',
                    },
                'tmean': {
                    'scale': 0.1,
                    'unit': cf_units.Unit('degrees Celsius'),
                    'long_name': 'mean temperature',
                    },
                'tmin': {
                    'scale': 0.1,
                    'unit': cf_units.Unit('degrees Celsius'),
                    'long_name': 'minimum temperature',
                    }
                }

        year_pattern = re.compile(r'_(\d{4})_')
        month_pattern = re.compile(r'_(\d{2})_')

        time_unit_str = 'hours since 1970-01-01 00:00:00'
        calendar = 'gregorian'
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        commit_hashes = set()
        cube_list = iris.cube.CubeList([])

        def update_hashes(commit_hash):
            commit_hashes.update([commit_hash])
            # TODO: Need to reinstate this constraint!!!!
            '''
            assert len(commit_hashes) == 1, (
                    "All loaded data should be from the same commit.")
            '''

        for f in files[process_slice]:
            # If this file has been regridded already and saved as a NetCDF
            # file, then do not redo this.
            nc_file = f.replace('.tif', '.nc')
            try:
                cubes = self.read_data(nc_file)
            except:
                # Try again, removing a potentially corrupt file
                # beforehand.
                logging.exception("Read failed, recreating:'{:}'"
                                  .format(nc_file))
                cubes = None
                try:
                    os.remove(nc_file)
                except:
                    logging.exception("File did not exist:'{:}'"
                                      .format(nc_file))

            if cubes:
                update_hashes(cubes[0].attributes['commit'])
                cube_list.extend(cubes)
                continue

            try:
                with rasterio.open(f) as dataset:
                    pass
            except rasterio.RasterioIOError as e:
                logging.exception("Corrupted file.")
                # Try to download file again.
                url = f.replace(
                        os.path.join(DATA_DIR, 'CHELSA'),
                        'https://www.wsl.ch/lud/chelsa')

                command = ("curl --connect-timeout 20 -L -o {:} {:}"
                           .format(f, url))
                logging.debug('Executing:{:}'.format(command))
                os.system(command)

            with rasterio.open(f) as dataset:
                # NOTE: Since data is are stored as unsigned 16 bit
                # integers, with temperature (in degrees Celsius) scaled by
                # a factor x10, space can be saved by saving data in
                # float16 format.
                variable_key = os.path.split(os.path.split(f)[0])[1]
                assert dataset.count == 1, "There should only be one band."
                data = dataset.read(1).astype('float16')
                data = np.ma.MaskedArray(data * mapping[variable_key]['scale'],
                                         np.isinf(data), dtype=data.dtype)

                latitudes = iris.coords.DimCoord(
                        get_centres(np.linspace(
                            dataset.bounds.top,
                            dataset.bounds.bottom,
                            dataset.shape[0] + 1)),
                        standard_name='latitude',
                        units='degrees')
                longitudes = iris.coords.DimCoord(
                        get_centres(np.linspace(
                            dataset.bounds.left,
                            dataset.bounds.right,
                            dataset.shape[1] + 1)),
                        standard_name='longitude',
                        units='degrees')

            grid_coords = [
                (latitudes, 0),
                (longitudes, 1)
                ]

            split_f = os.path.split(f)[1]
            time_coord = iris.coords.DimCoord(
                    cf_units.date2num(
                        datetime(
                            int(year_pattern.search(split_f).group(1)),
                            int(month_pattern.search(split_f).group(1)),
                            1),
                        time_unit_str,
                        calendar),
                    standard_name='time',
                    units=time_unit)

            cube = iris.cube.Cube(
                    data, dim_coords_and_dims=grid_coords,
                    units=mapping[variable_key]['unit'],
                    var_name=variable_key,
                    long_name=mapping[variable_key]['long_name'],
                    aux_coords_and_dims=[(time_coord, None)])

            # Regrid cubes to the same lat-lon grid.
            # TODO: change lat and lon limits and also the number of points!!
            # Always work in 0.25 degree steps? From the same starting point?
            regrid_cube = regrid(cube)

            # Need to save as float64 or float32, choose float64 for future
            # interoperability.
            regrid_cube.data = regrid_cube.data.astype('float64')
            commit_hash = self.save_data(regrid_cube, nc_file)

            # If None is returned, then the file already exists and is not
            # being overwritten, which should not happen, as we check for
            # the existence of the file above, loading the data in that
            # case.
            assert commit_hash is not None, (
                "Data should have been loaded before, since the file exists.")
            update_hashes(commit_hash)
            cube_list.append(regrid_cube)

        # TODO: TEMPORARY, in order to allow merging of data from different
        # commits!!
        for cube in cube_list:
            del cube.attributes['commit']

        self.cubes = cube_list.merge()
        assert len(self.cubes) == 4, (
            "There should be 4 variables.")

        # If all the data has been processed, not just a subset.
        if process_slice == slice(None):
            self.write_cache()

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes.extract(iris.Constraint(
            time=lambda t: end > t.point > start))


class Copernicus_SWI(Dataset):
    """For primary analysis, it is advisable to use hpc
    (cx1_scipts/run_swi_script.sh) in order to process the daily nc files
    into monthly nc files as a series of jobs, which would take an
    incredibly long time and large amounts of RAM ontherwise (on the order
    of days).

    Once that script has been run, the resulting nc files can be used to
    easily construct a large iris Cube containing all the desired monthly
    data.

    """

    def __init__(self, process_slice=slice(None)):
        """Initialise the cubes.

        Args:
            process_slice (slice): Used to limit the loading/processing of
                raw daily .nc files. Slices resulting in single elements
                (eg. slice(i, i+1)) will select a MONTH of data. For
                example, this can be done with i being the PBS array job
                index (for example) to quickly generate all the required
                monthly .nc files from the daily files using array jobs on
                the hpc.

        """
        self.dir = os.path.join(DATA_DIR, 'Copernicus_SWI')
        monthly_dir = os.path.join(self.dir, 'monthly')

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        # The raw data is daily data, which has to be averaged to yield
        # monthly data.
        files = glob.glob(
            os.path.join(self.dir, '**', '*.nc'), recursive=True)

        daily_files = []
        monthly_files = []
        for f in files:
            if 'monthly' in f:
                monthly_files.append(f)
            else:
                daily_files.append(f)

        # Get times from the filenames, instead of having to load the cubes
        # and look at the time coordinate that way.
        pattern = re.compile(r'(\d{4})(\d{2})(\d{2})')
        datetimes = [datetime(*map(int, pattern.search(f).groups()))
                     for f in files]

        # Isolate the year and month of each file only, and only in the
        # times of the requested slice.
        year_months = sorted(list(set(
            [datetime(dt.year, dt.month, 1)
             for dt in datetimes])))[process_slice]

        start_year_month = year_months[0]
        end_year_month = year_months[-1] + relativedelta(months=+1)

        selected_daily_files = []
        selected_monthly_files = []

        # TODO: Avoid loading daily files if the corresponding monthly
        # files are present!!!

        for i, dt in enumerate(datetimes):
            if start_year_month <= dt < end_year_month:
                f = files[i]
                # Prevent loading monthly files into the daily file list
                # which will get processed into monthly data.
                if 'monthly' in f:
                    selected_monthly_files.append(f)
                else:
                    selected_daily_files.append(f)

        commit_hashes = set()
        monthly_cubes = iris.cube.CubeList([])

        def update_hashes(commit_hash):
            commit_hashes.update([commit_hash])
            # TODO: Need to reinstate this constraint!!!!
            '''
            assert len(commit_hashes) == 1, (
                    "All loaded data should be from the same commit.")
            '''

        # Process the daily files here first, then combine with the already
        # processed monthly data later. Processing involves regridding to a
        # 0.25 degree resolution and averaging over months.
        if selected_daily_files:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=(
                    "Skipping global attribute 'long_name': 'long_name' is "
                    "not a permitted attribute"))
                daily_cubes = load_cubes(selected_daily_files)

            for cube in daily_cubes:
                # Make metadata uniform so they can be concatenated.
                del cube.attributes['identifier']
                del cube.attributes['title']
                del cube.attributes['time_coverage_start']
                del cube.attributes['time_coverage_end']

            # Concatenate daily cubes into larger cubes with the same
            # information (but with longer time coordinates).
            raw_cubes = daily_cubes.concatenate()

            for i in range(len(raw_cubes)):
                raw_cubes[i] = regrid(raw_cubes[i])
                iris.coord_categorisation.add_month_number(raw_cubes[i],
                                                           'time')
                iris.coord_categorisation.add_year(raw_cubes[i], 'time')
                monthly_cubes.append(raw_cubes[i].aggregated_by(
                    ['month_number', 'year'], iris.analysis.MEAN))

                # Save these monthly files separately.
                dt = raw_cubes[i].coord('time').cell(0).point
                year, month, day = dt.year, dt.month, dt.day
                commit_hash = self.save_data(
                        raw_cubes[i],
                        os.path.join(
                            monthly_dir,
                            ("c_gls_SWI_{:04d}{:02d}{:02d}_monthly"
                             "_GLOBE_ASCAT_V3.1.1.nc").format(
                                 year, month, day)))

                # If None is returned, then the file already exists and is not
                # being overwritten, which should not happen, as we check for
                # the existence of the file above, loading the data in that
                # case.
                assert commit_hash is not None, (
                    "Data should have been loaded before, "
                    "since the file exists.")
                update_hashes(commit_hash)

        if selected_monthly_files:
            monthly_cubes.extend(load_cubes(selected_monthly_files))

        # TODO: TEMPORARY, in order to allow merging of data from different
        # commits!!
        for cube in monthly_cubes:
            del cube.attributes['commit']

        # TODO: Verify that this works as expected.
        self.cubes = monthly_cubes.concatenate()

        # TODO: Caching!! Probably need to iterate over months beforehand
        # and do selective caching just like for CHELSA data.

        # If all the data has been processed, not just a subset.
        if process_slice == slice(None):
            self.write_cache()

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes.extract(iris.Constraint(
            time=lambda t: end > t.point > start))


class CRU(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'CRU')
        # Ignore warning regarding cloud cover units - they are fixed below.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=(
                "Ignoring netCDF variable 'cld' invalid units 'percentage'"))

            # TODO: In order to use the 'stn' variable - with information
            # about the measurement stations, the files have to be handled
            # individually so that we can keep track of which stn cube
            # belongs to which data cube.
            self.cubes = iris.load(glob.glob(os.path.join(self.dir, '*.nc')))

        # TODO: For now, remove the 'stn' cubes (see above).
        self.cubes = iris.cube.CubeList(
                [cube for cube in self.cubes if cube.name() != 'stn'])

        # Fix units for cloud cover.
        for cube in self.cubes:
            if cube.name() == 'cloud cover':
                cube.units = cf_units.Unit('percent')
                break

        # NOTE: Measurement times are listed as being in the middle of the
        # month, requiring no further intervention.

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes.extract(iris.Constraint(
            time=lambda t: end > t.point > start))


class ESA_CCI_Fire(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-Fire_burnedarea')
        self.cubes = iris.cube.CubeList([iris.load_cube(os.path.join(
                self.dir, 'MODIS_cci.BA.2001.2016.1440.720.365days.sum.nc'))])

    def get_monthly_data(self):
        raise NotImplementedError("Only yearly data available!")


class ESA_CCI_Landcover(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-LC_landcover',
                                '0d25_landcover')

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        filenames = glob.glob(os.path.join(self.dir, '*.nc'))
        filenames.sort()  # increasing years
        self.raw_cubes = iris.load(filenames)

        # To concatenate the cubes, take advantage of the fact that there
        # are 17 cubes per year, and then simply loop over the years,
        # joining the corresponding cubes into lists corresponding to their
        # variable.
        cube_lists = []
        for i in range(17):
            cube_lists.append(iris.cube.CubeList([]))

        n_years = len(self.raw_cubes) / 17
        assert np.isclose(n_years, int(n_years))
        n_years = int(n_years)

        years = range(1992, 2016)
        assert len(years) == n_years

        time_unit_str = 'hours since 1970-01-01 00:00:00'
        calendar = 'gregorian'
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        for i in range(n_years):
            time = iris.coords.DimCoord(
                    [cf_units.date2num(datetime(years[i], 1, 1),
                                       time_unit_str, calendar)],
                    standard_name='time',
                    units=time_unit)
            for j in range(17):
                cube = self.raw_cubes[(17*i) + j]

                cube_coords = cube.coords()

                cube2 = iris.cube.Cube(cube.lazy_data().reshape(1, 720, 1440))
                cube2.attributes = cube.attributes
                cube2.long_name = cube.long_name
                cube2.name = cube.name
                cube2.standard_name = cube.standard_name
                cube2.units = cube.units
                cube2.var_name = cube.var_name

                for key in ['id', 'tracking_id', 'date_created']:
                    del cube2.attributes[key]
                cube2.attributes['time_coverage_start'] = (
                        self.raw_cubes[0].attributes['time_coverage_start'])
                cube2.attributes['time_coverage_end'] = (
                        self.raw_cubes[-1].attributes['time_coverage_end'])

                cube2.add_dim_coord(time, 0)
                cube2.add_dim_coord(cube_coords[0], 1)
                cube2.add_dim_coord(cube_coords[1], 2)

                cube_lists[j].append(cube2)

        self.cubes = iris.cube.CubeList([])
        for cube_list in cube_lists:
            self.cubes.append(cube_list.concatenate_cube())

        self.write_cache()

    def get_monthly_data(self):
        raise NotImplementedError("Only yearly data available!")


class ESA_CCI_Landcover_PFT(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-LC_landcover',
                                '0d25_lc2pft')
        self.cubes = iris.load(glob.glob(os.path.join(self.dir, '*.nc')))
        time_coord = self.cubes[-1].coords()[0]
        assert time_coord.standard_name == 'time'

        # fix peculiar 'z' coordinate, which should be the number of years
        for cube in self.cubes:
            coord_names = [coord.name() for coord in cube.coords()]
            if 'z' in coord_names:
                assert coord_names[0] == 'z'
                cube.remove_coord('z')
                cube.add_dim_coord(time_coord, 0)

    def get_monthly_data(self):
        raise NotImplementedError("Only yearly data available!")


class ESA_CCI_Soilmoisture(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'ESA-CCI-SM_soilmoisture')
        self.cubes = iris.load(glob.glob(os.path.join(self.dir, '*.nc')))

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        # First get the desired cube from the list, then select the desired
        # timespan.
        return (self.cubes.extract_strict(
            iris.Constraint(name='Volumetric Soil Moisture Monthly Mean'))
            .extract(iris.Constraint(time=lambda t: end >= t.point >= start)))


class ESA_CCI_Soilmoisture_Daily(Dataset):

    def __init__(self):
        raise Exception("Use ESA_CCI_Soilmoisture Dataset for monthly data!")
        self.dir = os.path.join(DATA_DIR, 'soil-moisture', 'daily_files',
                                'COMBINED')
        files = sorted(glob.glob(os.path.join(
                self.dir, '**', '*.nc')))
        raw_cubes = load_cubes(files, 100)

        # Delete varying attributes.
        for cube in raw_cubes:
            for attr in ['id', 'tracking_id', 'date_created']:
                del cube.attributes[attr]

        # For the observation timestamp cubes, remove the 'valid_range'
        # attribute, which varies from cube to cube. The values of this
        # parameter are [-0.5, 0.5] for day 0, [0.5, 1.5] for day 1, etc...
        #
        # TODO: This seems to work but seems kind of hacky - is it really
        # guaranteed that the ordering of the cubes is constant?
        for cube in raw_cubes[7:None:8]:
            del cube.attributes['valid_range']

        self.cubes = raw_cubes.concatenate()

        for cube in self.cubes:
            iris.coord_categorisation.add_month_number(cube, 'time')
            iris.coord_categorisation.add_year(cube, 'time')

        # Perform averaging over months in each year.
        self.monthly_means = iris.cube.CubeList()
        for cube in self.cubes:
            self.monthly_means.append(cube.aggregated_by(
                ['month_number', 'year'],
                iris.analysis.MEAN))

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        # TODO: Isolate actual soil moisture.
        return self.monthly_means.extract(iris.Constraint(
            time=lambda t: end >= t.point >= start))


class GFEDv4(Dataset):
    """Without small fires.

    """
    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'gfed4', 'data')

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        filenames = glob.glob(os.path.join(self.dir, '*MQ*.hdf'))
        filenames.sort()  # increasing months & years

        datetimes = []
        data = []

        for f in filenames:
            hdf = SD(f, SDC.READ)
            # TODO: Use 'BurnedAreaUncertainty' dataset, and maybe others,
            # like 'FirePersistence' (viewed using hdf.datasets()).
            burned_area = hdf.select('BurnedArea')

            attributes = burned_area.attributes()

            split_f = os.path.split(f)[1]
            year = int(split_f[11:15])
            month = int(split_f[15:17])

            assert 1990 < year < 2030
            assert 0 < month < 13

            datetimes.append(datetime(year, month, 1))
            data.append(
                    burned_area[:][np.newaxis].astype('float64')
                    * attributes['scale_factor'])

        data = np.vstack(data)

        unit = cf_units.Unit(attributes['units'])
        long_name = attributes['long_name']

        calendar = 'gregorian'
        time_unit_str = 'days since 1970-01-01 00:00:00'
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)
        time_coord = iris.coords.DimCoord(
                [cf_units.date2num(dt, time_unit_str, calendar)
                 for dt in datetimes],
                standard_name='time',
                units=time_unit)

        latitudes = iris.coords.DimCoord(
                get_centres(np.linspace(90, -90, 721)),
                standard_name='latitude',
                units='degrees')
        longitudes = iris.coords.DimCoord(
                get_centres(np.linspace(-180, 180, 1441)),
                standard_name='longitude',
                units='degrees')

        latitudes.guess_bounds()
        longitudes.guess_bounds()

        burned_area_cube = iris.cube.Cube(
                data,
                long_name=long_name,
                units=unit,
                dim_coords_and_dims=[
                    (time_coord, 0),
                    (latitudes, 1),
                    (longitudes, 2)
                    ])

        # Normalise using the areas, divide by 10000 to convert from m2 to
        # hectares (the burned areas are in hectares originally).
        # NOTE: Some burned area percentages may be above 1!
        burned_area_cube.data /= (
                iris.analysis.cartography.area_weights(burned_area_cube)
                / 10000)
        burned_area_cube.units = cf_units.Unit('percent')

        self.cubes = iris.cube.CubeList([burned_area_cube])
        self.write_cache()

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes[0].extract(iris.Constraint(
            time=lambda t: end >= t.point >= start))


class GFEDv4s(Dataset):
    """Includes small fires.

    """

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'gfed4', 'data')

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

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
                datetime(year, month, 1),
                time_unit_str, calendar))

        time_coord = iris.coords.DimCoord(
                num_times, standard_name='time',
                units=time_unit)

        for coord in (longitudes, latitudes, time_coord):
            coord.guess_bounds()

        self.cubes = iris.cube.CubeList([iris.cube.Cube(
                np.vstack(data), dim_coords_and_dims=[
                    (time_coord, 0),
                    (latitudes, 1),
                    (longitudes, 2)
                    ])])

        self.cubes[0].units = cf_units.Unit('percent')
        self.cubes[0].var_name = 'Burnt_Area'
        self.write_cache()

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes[0].extract(iris.Constraint(
            time=lambda t: end >= t.point >= start))


class GlobFluo_SIF(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'GlobFluo_SIF')
        self.cubes = iris.cube.CubeList(
                [iris.load_cube(glob.glob(os.path.join(self.dir, '*.nc')))])

        # Need to convert to time coordinate, as values are relative to
        # 1582-10-14, which is not supported by the cf_units gregorian
        # calendar (needs to start from 1582-10-15, I think).

        # Get the original number of days relative to 1582-10-14 00:00:00.
        days_since_1582_10_14 = self.cubes[0].coords()[0].points
        # Define new time unit relative to a supported date.
        new_time_unit = cf_units.Unit('days since 1582-10-16 00:00:00',
                                      calendar='gregorian')
        # The corresponding number of days for the new time unit.
        days_since_1582_10_16 = days_since_1582_10_14 - 2

        self.cubes[0].remove_coord('time')
        new_time = iris.coords.DimCoord(
                days_since_1582_10_16,
                standard_name='time',
                units=new_time_unit)
        self.cubes[0].add_dim_coord(new_time, 0)

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes[0].extract(iris.Constraint(
            time=lambda t: end >= t.point >= start))


class GPW_v4_pop_dens(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'GPW_v4_pop_dens')
        netcdf_dataset = netCDF4.Dataset(glob.glob(
            os.path.join(self.dir, '*.nc'))[0])
        data = netcdf_dataset['Population Density, v4.10 (2000, 2005, 2010,'
                              ' 2015, 2020): 30 arc-minutes']

        datetimes = [datetime(year, 1, 1) for year in
                     [2000, 2005, 2010, 2015, 2020]]
        time_unit_str = 'days since {:}'.format(
                str(datetime(2000, 1, 1)))
        time_unit = cf_units.Unit(time_unit_str, calendar='gregorian')
        time = iris.coords.DimCoord(
                cf_units.date2num(datetimes, time_unit_str,
                                  calendar='gregorian'),
                standard_name='time',
                units=time_unit)

        latitudes = iris.coords.DimCoord(
                netcdf_dataset['latitude'][:], standard_name='latitude',
                units='degrees')
        longitudes = iris.coords.DimCoord(
                netcdf_dataset['longitude'][:], standard_name='longitude',
                units='degrees')

        coords = [
                (time, 0),
                (latitudes, 1),
                (longitudes, 2)
                ]

        self.cubes = iris.cube.CubeList([iris.cube.Cube(
                data[:5],
                long_name=data.long_name,
                var_name='Population_Density',
                units=cf_units.Unit('1/km2'),
                dim_coords_and_dims=coords)])

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        """Linear interpolation onto the target months.

        """
        start_month = start.month
        end_month = end.month
        start_year = start.year
        end_year = end.year

        year = start_year
        month = start_month

        datetimes = []
        while ((year != end_year) or (month != end_month)):
            datetimes.append(datetime(year, month, 1))

            month += 1
            if month == 13:
                month = 1
                year += 1

        # This is needed to include end month
        datetimes.append(datetime(year, month, 1))

        time_unit_str = 'days since {:}'.format(
                str(datetime(2000, 1, 1)))
        time_unit = cf_units.Unit(time_unit_str, calendar='gregorian')
        time = iris.coords.DimCoord(
                cf_units.date2num(datetimes, time_unit_str,
                                  calendar='gregorian'),
                standard_name='time',
                units=time_unit)

        interp_cubes = iris.cube.CubeList()
        for i in range(time.points.size):
            interp_points = [
                    ('time', time[i].points),
                    ]
            interp_cubes.append(
                    self.cubes[0].interpolate(
                        interp_points,
                        iris.analysis.Linear()))

        final_cubelist = interp_cubes.concatenate()
        assert len(final_cubelist) == 1
        final_cube = final_cubelist[0]
        return final_cube


class GSMaP_precipitation(Dataset):

    def __init__(self, times='00Z-23Z'):
        self.dir = os.path.join(
                DATA_DIR, 'GSMaP_Precipitation', 'hokusai.eorc.jaxa.jp',
                'realtime_ver', 'v6', 'daily_G', times)

        self.cubes = self.read_cache()
        # If a CubeList has been loaded successfully, exit __init__
        if self.cubes:
            return

        # Sort so that time is increasing.
        filenames = sorted(glob.glob(os.path.join(self.dir, '**', '*.nc')))

        calendar = 'gregorian'
        time_unit_str = 'days since 1970-01-01 00:00:00'
        time_unit = cf_units.Unit(time_unit_str, calendar=calendar)

        monthly_average_cubes = iris.cube.CubeList([])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=(
                "Collapsing a non-contiguous coordinate. Metadata may not "
                "be fully descriptive for 'time'."))
            for f in filenames:
                # Clip outer values which are duplicated in the data
                # selection below and not needed here.
                raw_cube = iris.load_cube(f)[..., 1:1441]
                monthly_cube = raw_cube.collapsed('time',
                                                  iris.analysis.MEAN)

                longitude_points = monthly_cube.coord('longitude').points
                assert np.min(longitude_points) == 0.125
                assert np.max(longitude_points) == 359.875

                # Modify the time coordinate such that it is recorded with
                # respect to a common date, as opposed to relative to the
                # beginning of the respective month as is the case for the
                # cube loaded above.
                centre_datetime = monthly_cube.coord('time').cell(0).point
                new_time = cf_units.date2num(centre_datetime,
                                             time_unit_str, calendar)
                monthly_cube.coord('time').bounds = None
                monthly_cube.coord('time').points = [new_time]
                monthly_cube.coord('time').units = time_unit

                monthly_average_cubes.append(monthly_cube)

        merged_cube = monthly_average_cubes.merge_cube()
        merged_cube.units = cf_units.Unit('mm/hr')

        # The cube still has lazy data at this point. So accessing the
        # 'data' attribute will involve averaging data and concatenating
        # it, which will take much longer than anything that is being
        # achieved above!

        self.cubes = iris.cube.CubeList([merged_cube])
        self.write_cache()

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes[0].extract(iris.Constraint(
            time=lambda t: end >= t.point >= start))


class LIS_OTD_lightning_climatology(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'LIS_OTD_lightning_climatology')
        self.cubes = iris.cube.CubeList(
                [iris.load(glob.glob(os.path.join(self.dir, '*.nc')))
                 .extract_strict(iris.Constraint(
                     name='Combined Flash Rate Monthly Climatology'))])

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        """'Broadcast' monthly climatology across the requested time
        period.

        NOTE: Not 'True' monthly data!!

        This method ignores days.

        """
        # TODO: Make this work with lazy data?

        cube = self.cubes[0]

        start_month = start.month
        end_month = end.month
        start_year = start.year
        end_year = end.year

        year = start_year
        month = start_month

        output_arrs = []
        datetimes = []

        while ((year != end_year) or (month != end_month)):
            output_arrs.append(cube[..., (month - 1)].data[np.newaxis])
            datetimes.append(datetime(year, month, 1))

            month += 1
            if month == 13:
                month = 1
                year += 1

        # This is needed to include end month
        output_arrs.append(cube[..., (month - 1)].data[np.newaxis])
        datetimes.append(datetime(year, month, 1))

        output_data = np.vstack(output_arrs)

        time_unit_str = 'days since {:}'.format(
                str(datetime(start_year, start_month, 1)))
        time_unit = cf_units.Unit(time_unit_str, calendar='gregorian')
        time_coord = iris.coords.DimCoord(
                cf_units.date2num(datetimes, time_unit_str,
                                  calendar='gregorian'),
                standard_name='time',
                units=time_unit)

        new_coords = [
                (time_coord, 0),
                (cube.coords()[0], 1),
                (cube.coords()[1], 2)
                ]

        output_cube = iris.cube.Cube(
                output_data,
                dim_coords_and_dims=new_coords,
                standard_name=cube.standard_name,
                long_name=cube.long_name,
                var_name=cube.var_name,
                units=cube.units,
                attributes=cube.attributes)

        return output_cube


class LIS_OTD_lightning_time_series(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'LIS_OTD_lightning_time_series')

        self.cubes = self.read_cache()
        # Exit __init__ if we have loaded the data.
        if self.cubes:
            return

        # Otherwise keep loading the data.
        raw_cubes = iris.load(glob.glob(os.path.join(self.dir, '*.nc')))
        # TODO: Use other attributes as well? Eg. separate LIS / OTD data,
        # grid cell area, or Time Series Sampling (km^2 / day)?

        # Isolate single combined flash rate.
        raw_cubes = raw_cubes.extract(
                iris.Constraint(name='Combined Flash Rate Time Series'))

        for cube in raw_cubes:
            iris.coord_categorisation.add_month_number(cube, 'time')
            iris.coord_categorisation.add_year(cube, 'time')

        monthly_cubes = [cube.aggregated_by(['month_number', 'year'],
                                            iris.analysis.MEAN)
                         for cube in raw_cubes]

        # Create new cube(s) where the time dimension is the first
        # dimension. To do this, the cube metadata can be copied, while new
        # coordinates and corresponding data (both simply
        # reshaped/reordered) are assigned.

        new_coords = [
                (monthly_cubes[0].coord('time'), 0),
                (monthly_cubes[0].coord('latitude'), 1),
                (monthly_cubes[0].coord('longitude'), 2)
                ]

        self.cubes = iris.cube.CubeList([])
        for cube in monthly_cubes:
            # NOTE: This does not use any lazy data whatsoever, starting
            # with the monthly aggregation above.
            assert cube.shape[-1] == len(cube.coord('time').points), (
                    "Old and new time dimension should have the same length")
            data_arrs = []
            for time_index in range(cube.shape[-1]):
                data_arrs.append(cube.data[..., time_index][np.newaxis])

            new_data = np.ma.vstack(data_arrs)

            new_cube = iris.cube.Cube(
                    new_data,
                    dim_coords_and_dims=new_coords)
            new_cube.metadata = deepcopy(cube.metadata)
            self.cubes.append(new_cube)

        self.write_cache()

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes.extract(iris.Constraint(
            time=lambda t: end > t.point > start))


class Liu_VOD(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'Liu_VOD')
        self.cubes = iris.cube.CubeList([iris.load_cube(
            glob.glob(os.path.join(self.dir, '*.nc')))])

        # Need to convert to time coordinate, as values are relative to
        # 1582-10-14, which is not supported by the cf_units gregorian
        # calendar (needs to start from 1582-10-15, I think).

        # Get the original number of days relative to 1582-10-14 00:00:00.
        days_since_1582_10_14 = self.cubes[0].coords()[0].points
        # Define new time unit relative to a supported date.
        new_time_unit = cf_units.Unit('days since 1582-10-16 00:00:00',
                                      calendar='gregorian')
        # The corresponding number of days for the new time unit.
        days_since_1582_10_16 = days_since_1582_10_14 - 2

        self.cubes[0].remove_coord('time')
        new_time = iris.coords.DimCoord(
                days_since_1582_10_16,
                standard_name='time',
                units=new_time_unit)
        self.cubes[0].add_dim_coord(new_time, 0)

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        return self.cubes[0].extract(iris.Constraint(
            time=lambda t: end >= t.point >= start))


class MOD15A2H_LAI_fPAR(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'MOD15A2H_LAI-fPAR')
        self.cubes = iris.cube.CubeList(
                [iris.load_cube(glob.glob(os.path.join(self.dir, '*.nc')))])
        assert len(self.cubes) == 1

        months = []
        for i in range(self.cubes[0].shape[0]):
            months.append(self.cubes[0].coords()[0].cell(i).point.month)

        assert np.all(np.diff(np.where(np.diff(months) != 1)) == 12), (
                "The year should increase every 12 samples!")

    def get_monthly_data(self, start=PartialDateTime(2000, 1),
                         end=PartialDateTime(2000, 12)):
        # TODO: Since the day in the month for which the data is provided
        # is variable, take into account neighbouring months as well in a
        # weighted average (depending on how many days away from the middle
        # of the month these other samples are)?
        return self.cube[0].extract(iris.Constraint(
            time=lambda t: end >= t.point >= start))


class Simard_canopyheight(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'Simard_canopyheight')
        self.cubes = iris.cube.CubeList(
                [iris.load_cube(glob.glob(os.path.join(self.dir, '*.nc')))])

    def get_monthly_data(self):
        raise NotImplementedError(
            "No time-varying information, only lat-lon values available.")


class Thurner_AGB(Dataset):

    def __init__(self):
        self.dir = os.path.join(DATA_DIR, 'Thurner_AGB')
        # Ignore warning about units, which are fixed below.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=(
                "Ignoring netCDF variable 'biomass\_totalag' invalid units"
                " 'kg\[C\]\/m2'"))
            warnings.filterwarnings("ignore", message=(
                "Ignoring netCDF variable 'biomass\_branches' invalid units"
                " 'kg\[C\]\/m2'"))
            warnings.filterwarnings("ignore", message=(
                "Ignoring netCDF variable 'biomass\_foliage' invalid units"
                " 'kg\[C\]\/m2'"))
            warnings.filterwarnings("ignore", message=(
                "Ignoring netCDF variable 'biomass\_roots' invalid units"
                " 'kg\[C\]\/m2'"))
            warnings.filterwarnings("ignore", message=(
                "Ignoring netCDF variable 'biomass\_stem' invalid units"
                " 'kg\[C\]\/m2'"))

            self.cubes = iris.load(glob.glob(os.path.join(self.dir, '*.nc')))

        for cube in self.cubes:
            cube.units = cf_units.Unit('kg(C)/m2')

    def get_monthly_data(self):
        raise NotImplementedError("Data is static.")


def load_dataset_cubes():
    a = CRU()
    b = ESA_CCI_Soilmoisture()
    c = GFEDv4s()
    d = GlobFluo_SIF()
    e = Liu_VOD()

    # Join up all the cubes.
    cubes = iris.cube.CubeList()
    cubes.extend(a.cubes)
    # For now, only use the monthly mean soil moisture data.
    cubes.append(b.cubes.extract_strict(
            iris.Constraint(name='Volumetric Soil Moisture Monthly Mean')))
    cubes.extend(c.cubes)
    cubes.extend(d.cubes)
    cubes.extend(e.cubes)

    min_times = [c.coords()[0].cell(0).point for c in cubes]
    max_times = [c.coords()[0].cell(-1).point for c in cubes]

    # This timespan will encompass all the datasets.
    min_time = np.max(min_times)
    max_time = np.min(max_times)

    # This method returns a cube.
    f = LIS_OTD_lightning_climatology().get_monthly_data(min_time, max_time)
    assert isinstance(f, iris.cube.Cube)
    cubes.append(f)

    g = GPW_v4_pop_dens().get_monthly_data(min_time, max_time)
    assert isinstance(g, iris.cube.Cube)
    cubes.append(g)

    # Extract the common timespan.
    # t is a Cell, t.point extracts the 'real_datetime' object which has
    # the expected year and month attributes. This also leads to ignoring
    # bounds, which, if they are not None, can cause these comparisons to
    # fail.
    cubes = cubes.extract(iris.Constraint(
        time=lambda t: monthly_constraint(t.point, (min_time, max_time))))

    # Regrid cubes to the same lat-lon grid.
    # TODO: change lat and lon limits and also the number of points!!
    # Always work in 0.25 degree steps? From the same starting point?
    for i in range(len(cubes)):
        cubes[i] = regrid(
            cubes[i],
            area_weighted=True,
            new_latitudes=get_centres(np.linspace(-90, 90, 361)),
            new_longitudes=get_centres(np.linspace(-180, 180, 721)))

    return cubes


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # a = CHELSA()
    # a = GFEDv4()
    a = Copernicus_SWI(slice(0, 3))


if __name__ == '__main__2':
    logging.basicConfig(level=logging.INFO)

    # TODO: Use iris cube long_name attribute to enter descriptive name
    # which will be used throughout data analysis and plotting (eg. for
    # selecting DataFrame columns).

    # agb = AvitabileThurnerAGB()
    # plt.close('all')
    # plt.figure()
    # qplt.contourf(agb.cube, 20)
    # plt.gca().coastlines()

    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            cubes = pickle.load(f)
    else:
        cubes = load_dataset_cubes()
        with open(pickle_file, 'wb') as f:
            pickle.dump(cubes, f, -1)

    # Use masking to extract only the relevant data.

    # Accumulate the masks for each dataset into a global mask.
    global_mask = np.zeros(cubes[0].shape, dtype=bool)
    for cube in cubes:
        global_mask |= combine_masks(cube.data, invalid_values=[])

    # Apply the same mask for each latitude and longitude.
    collapsed_global_mask = np.any(global_mask, axis=0)
    global_mask = np.zeros_like(global_mask, dtype=bool)
    global_mask += collapsed_global_mask[np.newaxis]

    # Use this mask to select each dataset

    selected_datasets = []
    for cube in cubes:
        selected_data = cube.data[~global_mask]
        if hasattr(selected_data, 'mask'):
            assert not np.any(selected_data.mask)
            selected_datasets.append((cube.name(), selected_data.data))
        else:
            selected_datasets.append((cube.name(), selected_data))

    dataset_names = [s[0] for s in selected_datasets]

    exog_name_map = {
            'diurnal temperature range': 'diurnal temp range',
            'near-surface temperature minimum': 'near-surface temp min',
            'near-surface temperature': 'near-surface temp',
            'near-surface temperature maximum': 'near-surface temp max',
            'wet day frequency': 'wet day freq',
            'Volumetric Soil Moisture Monthly Mean': 'soil moisture',
            'SIF': 'SIF',
            'VODorig': 'VOD',
            'Combined Flash Rate Monthly Climatology': 'lightning rate',
            ('Population Density, v4.10 (2000, 2005, 2010, 2015, 2020)'
             ': 30 arc-minutes'): 'pop dens'
            }

    inclusion_names = {
            'near-surface temperature maximum',
            'Volumetric Soil Moisture Monthly Mean',
            'SIF',
            'VODorig',
            'diurnal temperature range',
            'wet day frequency',
            'Combined Flash Rate Monthly Climatology',
            ('Population Density, v4.10 (2000, 2005, 2010, 2015, 2020)'
             ': 30 arc-minutes'),
            }

    exog_names = [exog_name_map.get(s[0], s[0]) for s in
                  selected_datasets if s[0] in inclusion_names]
    raw_exog_data = np.hstack(
            [s[1].reshape(-1, 1) for s in selected_datasets
             if s[0] in inclusion_names])

    endog_name = selected_datasets[dataset_names.index('Burnt_Area')][0]
    endog_data = selected_datasets[dataset_names.index('Burnt_Area')][1]

    # lim = int(5e3)
    lim = None
    endog_data = endog_data[:lim]
    raw_exog_data = raw_exog_data[:lim]

    endog_data = pd.Series(endog_data, name='burned area')
    exog_data = pd.DataFrame(
            raw_exog_data,
            columns=exog_names)

    # TODO: Improve this by taking into account the number of days in each
    # month

    # Define dry days variable using the wet day variable.
    exog_data['dry day freq'] = 31.5 - exog_data['wet day freq']
    del exog_data['wet day freq']

    # Carry out log transformation for select variables.
    log_var_names = ['diurnal temp range',
                     # There are problems with negative surface
                     # temperatures here!
                     # 'near-surface temp max',
                     'dry day freq']

    for name in log_var_names:
        mod_data = exog_data[name] + 0.01
        assert np.all(mod_data > 0.01)
        exog_data['log ' + name] = np.log(mod_data)
        del exog_data[name]

    # Carry out square root transformation
    sqrt_var_names = ['lightning rate', 'pop dens']
    for name in sqrt_var_names:
        exog_data['sqrt ' + name] = np.sqrt(exog_data[name])
        del exog_data[name]

    '''
    # Available links for Gaussian:
    [statsmodels.genmod.families.links.log,
     statsmodels.genmod.families.links.identity,
     statsmodels.genmod.families.links.inverse_power]

    '''

    model = sm.GLM(endog_data, exog_data,
                   # family=sm.families.Gaussian(links.log)
                   family=sm.families.Binomial()
                   )

    model_results = model.fit()

    sm.graphics.plot_partregress_grid(model_results)
    plt.tight_layout()

    plt.figure()
    plt.hexbin(endog_data, model_results.fittedvalues, bins='log')
    plt.xlabel('real data')
    plt.ylabel('prediction')
