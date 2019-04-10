#!/usr/bin/env python3
# -*- coding: utf-8 -*_
"""Tools for downloading ERA5 data using the CDS API.

"""
from abc import ABC, abstractmethod
import calendar
from copy import deepcopy
from datetime import datetime
import json
import logging
import logging.config
from multiprocessing import Process, Pipe, Queue
import os
from queue import Empty
import sys
from threading import Thread
from time import time, sleep
import warnings

import cdsapi
from dateutil.relativedelta import relativedelta
import iris
from iris.time import PartialDateTime
import numpy as np

from wildfires.data.datasets import DATA_DIR
from wildfires.data.era5_tables import get_short_to_long
from wildfires.logging_config import LOGGING


logger = logging.getLogger(__name__)
variable_mapping = get_short_to_long()


def format_request(request):
    """Format the request tuple for nicer printing.

    Returns:
        str: Formatted request.

    """
    request = deepcopy(request)
    request_dict = request[1]
    days = []
    for year in request_dict['year']:
        for month in request_dict['month']:
            n_days = calendar.monthrange(int(year), int(month))[1]
            days.append(n_days)

    if len(request_dict['day']) == max(days):
        day_str = 'ALL'
    else:
        day_str = ', '.join(request_dict['day'])

    if len(request_dict['time']) == 24:
        time_str = 'ALL'
    else:
        time_str = ', '.join(request_dict['time'])

    if len(request_dict['month']) == 12:
        month_str = 'ALL'
    else:
        month_str = ', '.join(
            calendar.month_abbr[int(month)] for month in request_dict['month'])

    year_str = ', '.join(request_dict['year'])

    output = ("{} from {} for year(s) {}, month(s) {}, day(s) {}, time(s) {}."
              .format(request_dict['variable'], request[0],
                      year_str, month_str, day_str, time_str))
    return output


def str_to_seconds(s):
    """Pocesses a string including time units into a float in seconds.

    Args:
        s (str): Input string, including units.

    Returns:
        float: The processed time in seconds.

    Examples:
        >>> int(round(str_to_seconds('1')))
        1
        >>> int(round(str_to_seconds('1s')))
        1
        >>> int(round(str_to_seconds('2m')))
        120
        >>> int(round(str_to_seconds('3h')))
        10800
        >>> int(round(str_to_seconds('2d')))
        172800

    """
    if isinstance(s, str):
        multipliers = {
            's': 1.,
            'm': 60.,
            'h': 60.**2.,
            'd': 24. * 60**2.
            }
        for key, multiplier in zip(multipliers, list(multipliers.values())):
            if key in s:
                return float(s.strip(key)) * multiplier
    return float(s)


def format_variable(
        variable, variable_name, single_value_formatter=str,
        valid_single_types=(str, float, np.float, int, np.integer)):
    """Format a variable consistently.

    Args:
        variable (one of 'valid_single_types' or a list of
            'valid_single_types'): The variable to format.
        variable_name (str): The name of the variable. Only relevant for
            error messages.
        single_value_formatter (callable): Function called for each single
            variable in 'variable'.
        valid_single_types (tuple of types): Types which are handled
            correctly by the 'single_value_formatter' callable.

    Returns:
        list: A list (containing one or more elements) of the formatted
            variables.

    Raises:
        TypeError: If one of the elements of 'variable' do not match the
            'valid_single_types'.

    """

    if isinstance(variable, valid_single_types):
        return [single_value_formatter(variable)]

    type_error_msg = "Type {} not supported for argument '{}'".format(
        type(variable), variable_name)

    if hasattr(variable, '__iter__'):
        formatted_variables = []
        for single_value in variable:
            if not isinstance(single_value, valid_single_types):
                raise TypeError(type_error_msg)
            formatted_variables.append(single_value_formatter(single_value))
        return formatted_variables
    raise TypeError(type_error_msg)


def retrieve_hourly(variable='2m_temperature', levels='sfc', hours=None,
                    start=PartialDateTime(2000, 1, 1),
                    end=PartialDateTime(2000, 2, 1),
                    target_dir=os.path.join(DATA_DIR, 'ERA5'),
                    download=False):
    """Retrieve hourly ERA5 data for the chosen variable.

    Args:
        variable (str or list of str): Variable of interest: eg.
            variable='2t' or variable='2m_temperature' refers to
            the 2m surface (sfc) level temperature variable. Multiple
            variables are also possible if given as a list: eg.
            variable=['2t', '10u']  would retrieve 2m temperature and
            10 m U wind component.
        levels (str, int, or list of str or int): If levels='sfc', the
            surface data will be requested. Alternatively, levels='100',
            levels='100 hPa' or levels=100 would all select the 100 hPa
            level. Level values can also be put be given as a list.
        hours (None, str, int, or list of str or int): If hours=None,
            retrieve all hours. Alternatively hours may be given as
            integers (eg. hours=1), strings (eg. hours='01:00') or as lists
            of these. The hours must be in the range [0, 23].
        start (datetime): Initial datetime. This is inclusive (see 'end').
        end (datetime): Final datetime. This is not inclusive. So
            start=PartialDateTime(2000, 1, 1), end=PartialDateTime(2000, 2, 1)
            will retrieve all data for January.
        target_dir (str): Directory path where the output files will be stored.
        download (bool): If True, download data one requests at a time. If
            False, simply return the list of request tuples that can be
            used to download data.

    Returns:
        list: list of request tuples. Each tuple contains the dataset
            string, the request body as a dictionary, and the filename as a
            string. There will be one output filename per month containing
            all of the requested variables, named like
            era5_hourly_reanalysis_{year}_{month}.nc.

    Note:
        Reanalysis data is retrieved.

        Variables may have different names depending on whether the 'sfc'
        level or a pressure level is requested.

        Time information (ie. hours, minutes, etc...) in the start and end
        arguments will be ignored.

    """
    if download:
        client = cdsapi.Client(quiet=True)
    else:
        client = None

    if isinstance(variable, str):
        variable = [variable]

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    surface_level_dataset_id = 'reanalysis-era5-single-levels'
    pressure_level_dataset_id = 'reanalysis-era5-pressure-levels'

    if levels.lower() in ('sfc', 'surface'):
        logger.debug('Retrieving surface dataset.')
        dataset = surface_level_dataset_id
    else:
        logger.debug('Retrieving pressure level dataset.')
        dataset = pressure_level_dataset_id

    if hours is None:
        hours = ['{:02d}:00'.format(hour) for hour in range(0, 24)]
    else:
        def single_hour_formatter(hour):
            if isinstance(hour, str):
                if ':' in hour:
                    hour = hour[:hour.find(':')]
                else:
                    assert len(hour) <= 2, (
                        "'hours' written like {} are not supported."
                        .format(hour))
                hour = int(hour)
            elif isinstance(hour, (float, np.float)):
                hour = round(hour)
            # No else statement here is needed due to the types given to
            # the 'format_variable' function via 'valid_single_types'.
            return '{:02d}:00'.format(hour)

        hours = format_variable(hours, 'hours', single_hour_formatter,
                                (str, float, np.float, int, np.integer))
    logger.debug('Request hours:{}.'.format(hours))

    # 'levels' is only relevant for the pressure level dataset.
    if dataset == pressure_level_dataset_id:
        def single_level_formatter(level):
            if isinstance(level, str):
                # Remove 'hPa'. Trailing space would be stripped int().
                level = int(level.lower().strip('hpa'))
            elif isinstance(level, (float, np.float)):
                level = round(level)
            return str(level)

        levels = format_variable(levels, 'levels', single_level_formatter,
                                 (str, float, np.float, int, np.integer))
    logger.debug('Request levels:{}.'.format(levels))

    # Accumulate the date strings.
    monthly_dates = dict()
    start_date = datetime(start.year, start.month, start.day)
    current_date = start_date
    end_date = datetime(end.year, end.month, end.day)

    # Since the date given for 'end' is not inclusive.
    while current_date != end_date:
        current_key = (current_date.year, current_date.month)
        if current_key not in monthly_dates:
            monthly_dates[current_key] = {
                'year': [str(current_date.year)],
                'month': ['{:02d}'.format(current_date.month)],
                'day': ['{:02d}'.format(current_date.day)]
                }
        else:
            monthly_dates[current_key]['day'].append(
                '{:02d}'.format(current_date.day))
        current_date += relativedelta(days=+1)

    requests = []
    for request_date in monthly_dates.values():
        logger.debug('Request dates:{}.'.format(request_date))
        request_dict = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'year': request_date['year'],
            'month': request_date['month'],
            'day': request_date['day'],
            'time': hours
            }

        if dataset == pressure_level_dataset_id:
            request_dict['pressure_level'] = levels

        target_file = os.path.join(
            target_dir,
            'era5_hourly_reanalysis_{year}_{month}.nc'.format(
                year=request_dict['year'][0], month=request_dict['month'][0]))

        request = (dataset, request_dict, target_file)
        requests.append(request)
        if download:
            logger.info("Starting download to:'{}'.".format(target_file))
            client.retrieve(*request)
            logger.info("Finished download to:'{}'.".format(target_file))

    return requests


def retrieve_monthly(variable='167.128', start=PartialDateTime(2000, 1),
                     end=PartialDateTime(2000, 2), target_dir=DATA_DIR):
    """Retrieve monthly ERA5 data for the chosen variable.

    Args:
        variable (str): Variable of interest: eg. variable='167.128' refers to
            the 2m surface (sfc) level temperature variable. Multiple
            variables are also possible: eg. variable='167.128/207.128' would
            retrieve 2m temperature and 10m wind speed.
        start (datetime): Initial datetime. This is inclusive.
        end (datetime): Final datetime. This is inclusive.
        target (str): Directory path where the output files will be stored.

    Returns:
        list: list of output filenames.

    Note:
        Daily information present in the supplied dates is simply discarded.

        The queue for retrieving the pre-calculated monthly means (which
        this function relies on) has a far lower priority than the queue
        for hourly data, leading to very large retrieval times on the order
        of many hours, sometimes days for a single month!

    The output data are organised as one file per decade:
     - ...
     - 'era5_moda_{variable}_1990'
     - 'era5_moda_{variable}_2000'
     - ...

    """
    output_filenames = []
    client = cdsapi.Client(quiet=True)
    # ERA5 monthly data is timestamped to the first of the month, hence dates
    # have to be specified as a list in this format:
    # '19950101/19950201/19950301/.../20051101/20051201'.

    # Data is stored on one tape per decade, so for efficiency we split the
    # date range into decades, hence we get multiple date lists by decade:
    #  - '19950101/19950201/19950301/.../19991101/19991201'
    #  - '20000101/20000201/20000301/.../20051101/20051201'
    #  - '20000101/20000201/20000301/.../20051101/20051201'
    start = PartialDateTime(start.year, start.month)
    end = PartialDateTime(end.year, end.month)

    decades = sorted(list(np.array(
        list(set(year // 10 for year in range(start.year, end.year + 1))))
        * 10))

    for decade in decades:
        logger.debug('decade:{}'.format(decade))
        requested_dates = []
        # The decade is the first year, so go from there until the final
        # year, up to 9 times (otherwise the next decade would be reached).
        for year in range(decade, min((end.year, decade + 9)) + 1):
            logger.debug('year:{}'.format(year))
            for month in range(1, (end.month + 1) if year == end.year else 13):
                logger.debug('month:{}'.format(month))
                requested_dates.append('{:>04d}{:>02d}01'.format(year, month))

        date_request_string = '/'.join(requested_dates)
        decade_target_file = os.path.join(
            target_dir, 'era5_moda_{}_{}.nc'.format(variable, decade))

        logger.debug('date request:{}'.format(date_request_string))

        client.retrieve(
            "reanalysis-era5-complete",
            {
                'class': 'ea',
                'expver': '1',
                'stream': 'moda',
                'type': 'an',
                # 128 is table 2 version
                # https://confluence.ecmwf.int/display/UDOC/Identification+keywords
                'param': '167.128',
                'levtype': 'sfc',
                'date': date_request_string,
                'decade': str(decade), 'grid': '0.25/0.25',
                # Optional. Subset (clip) to an area. Specify as N/W/S/E in
                # Geographic lat/long degrees. Southern latitudes and
                # western longitudes must be given as negative numbers.
                # Requires "grid" to be set to a regular grid like "0.25/0.25".
                # 'area': '89.75/-179.75/-89.75/179.75',
                'format': 'netcdf'
            },
            decade_target_file)
        logger.info('Finished download for decade:{} to:{}'
                    .format(decade, decade_target_file))
        output_filenames.append(decade_target_file)

    return output_filenames


class RampVar:
    """Variable that is increased upon every call.

    The starting value, maximum value and the steps can be set.

    The value is incremented linearly between the initial and maximum
    value, with `steps` intervals.

    Args:
        initial_value (float): Initial value.
        max_value (float): Maximum value the variable can take.
        steps (int): The number of intervals.

    Examples:
        >>> var = RampVar(0, 2, 3)
        >>> int(round(var.value))
        0
        >>> int(round(var.value))
        1
        >>> int(round(var.value))
        2
        >>> var.reset()
        >>> int(round(var.value))
        0

    """

    def __init__(self, initial_value, max_value, steps=10):
        self.steps = steps
        self.values = np.linspace(initial_value, max_value, steps)
        self.index = -1

    @property
    def value(self):
        """Every time this attribute is accessed it is incremented as
        defined by the values given to the constructor.

        """
        if self.index < self.steps - 1:
            self.index += 1
        return self.values[self.index]

    def reset(self):
        """Resets the value to the initial value."""
        self.index = -1


class DownloadThread(Thread):
    """Retrieve data using the CDS API."""

    def __init__(self, id_index, request, queue):
        super().__init__()
        self.id_index = id_index
        self.request = request
        self.formatted_request = format_request(self.request)
        self.queue = queue

        # Configures a logger named after the class using the 'wildfires'
        # package logging configuration.
        self.logger_name = "{}.{:03d}".format(self.__class__.__name__,
                                              self.id_index)
        self.logger = logging.getLogger(self.logger_name)
        self.config_dict = deepcopy(LOGGING)
        self.config_dict['formatters']['default']['format'] = (
            self.config_dict['formatters']['default']['format'].replace(
                "%(message)s",
                "{:03d} | %(message)s".format(self.id_index)))
        orig_loggers = self.config_dict.pop('loggers')
        orig_loggers_dict = orig_loggers[list(orig_loggers.keys())[0]]
        self.config_dict['loggers'] = dict(
            ((self.logger_name, orig_loggers_dict),))
        logging.config.dictConfig(self.config_dict)

        # Need quiet=True, because otherwise the initialisation of Client
        # will call logging.basicConfig, which modifies the root logger and
        # results in duplicated logging messages with our current setup.
        self.client = cdsapi.Client(quiet=True)
        self.logger.debug("Initialised DownloadThread with id_index={}."
                          .format(self.id_index))

    @staticmethod
    def get_request_log_file(request):
        orig_dir, orig_filename = os.path.split(request[2])
        request_log_file = os.path.join(
                orig_dir, '.requests',
                '.'.join(orig_filename.split('.')[:-1] + ['request']))
        return request_log_file

    @staticmethod
    def retrieve_request(request):
        """Retrieve the original request associated with the target
        filename in the request.

        """
        try:
            request_log_file = DownloadThread.get_request_log_file(request)
            with open(request_log_file, 'r') as f:
                stored_request = tuple(json.load(f))
            return stored_request
        except FileNotFoundError:
            logger.error(
                "Request log could not be found for the following file: {}."
                .format(request[2]))
            return False

    def record_request(self):
        request_log_file = self.get_request_log_file(self.request)
        if not os.path.isdir(os.path.dirname(request_log_file)):
            os.makedirs(os.path.dirname(request_log_file))
        self.logger.debug("Recording request in file '{}'"
                          .format(request_log_file))
        with open(request_log_file, 'w') as f:
            json.dump(self.request, f, indent=4)

    def run(self):
        try:
            self.logger.info('Requesting: {}'.format(self.formatted_request))
            self.client.retrieve(*self.request)
            self.logger.debug('Completed request.')
            filename = self.request[2]
            if not os.path.isfile(filename):
                raise RuntimeError(
                    "Filename '{}' not found despite request "
                    "{} having being issued.".format(
                        filename, self.request))
            self.record_request()
            self.queue.put(self.request)
        except Exception:
            self.queue.put(sys.exc_info())
        finally:
            self.logger.debug("Exiting.")


class Worker(Process, ABC):
    """Abstract base class to subclass for use as a processing_worker in
    `retrieval_processing."""

    def __init__(self, id_index, pipe, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_index = id_index
        self.pipe = pipe

        # Configures a logger named after the class using the 'wildfires'
        # package logging configuration.
        self.logger_name = "{}.{:03d}".format(self.__class__.__name__,
                                              self.id_index)
        self.logger = logging.getLogger(self.logger_name)
        self.config_dict = deepcopy(LOGGING)
        self.config_dict['formatters']['default']['format'] = (
            self.config_dict['formatters']['default']['format'].replace(
                "%(message)s",
                "{:03d} | %(message)s".format(self.id_index)))
        orig_loggers = self.config_dict.pop('loggers')
        orig_loggers_dict = orig_loggers[list(orig_loggers.keys())[0]]
        self.config_dict['loggers'] = dict(
            ((self.logger_name, orig_loggers_dict),))
        logging.config.dictConfig(self.config_dict)

        self.logger.debug("Initialised {} with id_index={}."
                          .format(self.__class__.__name__, self.id_index))

    @abstractmethod
    def output_filename(self, input_filename):
        """Construct the output filename from the input filename."""
        pass

    @abstractmethod
    def check_output(self, request, output):
        """Check that the output matches the request.

        Args:
            request (iterable of str, dict, str): A request tuple as returned
                by `retrieve_hourly`.
            output (iterable of int, str, str): Output of `self.process`.

        Returns:
            bool: True if the output matches the request, False otherwise.

        """
        pass

    @abstractmethod
    def process(self, request):
        """Process data in the given request.

        Args:
            request (tuple): Request tuple as returned by `retrieve_monthly`.

        Returns:
            int: 0 for successful computation of the average. 1 is returned
                if an error is encountered. Note that exceptions are merely
                logged and not raised.
            str: The original filename `filename`.
            str or None: The output filename (if successful) or None.

        """
        pass

    def run(self):
        try:
            self.logger.debug("Started listening for filenames to process.")
            while True:
                request = self.pipe.recv()
                if request == "STOP_WORKER":
                    logger.debug(
                        "STOP_WORKER received, breaking out of loop.")
                    break
                file_to_process = request[2]
                self.logger.debug("Received file: '{}'. Starting processing."
                                  .format(file_to_process))
                output = self.process(request)
                self.logger.debug("Finished processing '{}' with status {}."
                                  .format(file_to_process, output))
                self.pipe.send(output)
                self.logger.debug("Sent status {}.".format(output))
        except Exception:
            self.pipe.send(sys.exc_info())
        finally:
            self.logger.debug("Exiting.")


class AveragingWorker(Worker):
    """Compute monthly averages using filenames passed in via a pipe."""

    def output_filename(self, input_filename):
        """Construct the output filename from the input filename."""
        return input_filename.split('.nc')[0] + '_monthly_mean.nc'

    def check_output(self, request, output):
        """Check that the output matches the request.

        Args:
            request (iterable of str, dict, str): A request tuple as returned
                by `retrieve_hourly`.
            output (iterable of int, str, str): Output of `self.process`.

        Returns:
            bool: True if the output matches the request, False otherwise.

        """
        if output[0] != 0:
            self.logger.warning(
                "Output is not as expected because processing of the "
                "request {} failed with error code {}"
                .format(request, output[0]))
            return False
        downloaded_file = request[2]
        output_file = output[2]
        expected_file = self.output_filename(downloaded_file)
        if output_file != expected_file:
            self.logger.warning(
                "Filenames do not match. Expected '{}', got '{}'."
                .format(expected_file, output_file))
            return False

        if not os.path.isfile(output_file):
            self.logger.warning("Expected output file '{}' does not exist."
                                .format(output_file))
            return False

        request_dict = request[1]
        years = list(map(int, request_dict['year']))
        months = list(map(int, request_dict['month']))
        days = list(map(int, request_dict['day']))
        hours = [int(time.replace(':00', '')) for time in request_dict['time']]

        datetime_range = (
            datetime(min(years), min(months), min(days), min(hours)),
            datetime(
                max(years), max(months), min((max(days),
                calendar.monthrange(max(years), max(months))[1])), max(hours)),
            )

        request_variables = request_dict['variable']
        # TODO: Can this be made more fine-grained to check each individual
        # expected name against the corresponding cube? This is impeded
        # by not knowing how the cubes are ordered with respect to the
        # ordering of variables in the original request.
        # NOTE: The current approach achieves the same thing since matches
        # are consumed, but it is more cumbersome.
        expected_name_sets = []
        for variable in request_variables:
            expected_name_sets.append({
                variable,
                variable_mapping[variable]
            })
        try:
            output_cubes = iris.load(output_file)
        except Exception:
            self.logger.exception("Error while loading '{}'."
                                  .format(output_file))
            return False

        # Each cube will have one variable.
        for cube in output_cubes:
            bounds = cube.coord('time').cell(0).bound
            which_failed = []
            error_details = []
            if bounds != datetime_range:
                which_failed.append("bounds check")
                error_details.append("Expected bounds {}, got bounds {}."
                                     .format(bounds, datetime_range))
            raw_cube_names = (
                cube.standard_name,
                cube.long_name,
                cube.var_name
                )
            cube_names = list(map(str, raw_cube_names))

            passed_var_check = False
            for name_index, expected_name_set in enumerate(expected_name_sets):
                if expected_name_set.intersection(cube_names):
                    logger.debug("Matched {} with {}"
                                 .format(expected_name_set, cube_names))
                    passed_var_check = True
                    # Next time, there will be one fewer variable to compare
                    # against.
                    del expected_name_sets[name_index]
                    break
            if not passed_var_check:
                which_failed.append("variable name check")
                error_details.append(
                    "None of {} matched one of the expected names {}."
                    .format(
                        ', '.join(cube_names),
                        ', '.join(
                            [name
                             for name_set in expected_name_sets
                             for name in name_set])))

            if which_failed:
                which_failed = ' and '.join(which_failed)
                error_details = ' '.join(error_details)
                self.logger.warning(
                    "Failed {} for cube {}. {}"
                    .format(which_failed, repr(cube), error_details))
                return False
        return True

    def process(self, request):
        """Performs monthly averaging on the data in the given request.

        Args:
            request (tuple): Request tuple as returned by `retrieve_monthly`.

        Returns:
            int: 0 for successful computation of the average. 1 is returned
                if an error is encountered. 2 is returned if no exception
                was encountered, but the resulting data does not match the
                expectations as defined by `self.check_output`. Note that
                exceptions are merely logged and not raised.
            str: The original filename `filename`.
            str or None: The output filename (if successful) or None.

        """
        filename = request[2]
        self.logger.debug("Processing: '{}'.".format(filename))
        try:
            cubes = iris.load(filename)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=(
                    "Collapsing a non-contiguous coordinate. Metadata may not "
                    "be fully descriptive for 'time'."))
                cubes = iris.cube.CubeList(
                    [cube.collapsed('time', iris.analysis.MEAN)
                     for cube in cubes])
                # TODO: Verify that dask arrays are indeed not being saved.
                # Realise data so we don't end up storing dask arrays
                # instead of numpy arrays.
                # [cube.data for cube in cubes]
            save_name = self.output_filename(filename)
            iris.save(cubes, save_name, zlib=False)
            # If everything went well.
            if self.check_output(request, (0, filename, save_name)):
                return (0, filename, save_name)
            return (2, filename, None)
        except Exception:
            self.logger.exception("Error while processing '{}'."
                                  .format(filename))
            return (1, filename, None)


def retrieval_processing(requests, processing_class=AveragingWorker,
                         n_threads=4, soft_filesize_limit=1000,
                         timeout='3d', delete_processed=True,
                         overwrite=False):
    """Start retrieving and processing data asynchronously.

    The calling process spawns one non-blocking process just for the
    processing of the downloaded files. This process is fed with the
    filenames of the files as they are downloaded by the download threads.
    It then proceeds to average these values (without blocking the download
    of the other data or blocking).

    A number of threads (`n_threads`) will be started in order to download
    data concurrently. If one thread finishes downloading data, it will
    receive a new request to retrieve, and so on for the other threads,
    until all the requests have been handled.

    The main process checks the output of these download threads, and if a
    new file has been downloaded successfully, it is added to the
    processing queue for the distinct processing process.

    Note:
        Using NetCDF and a 0.25 x 0.25 grid, each variable takes up ~1.5 GB
        per pressure level per month.

    Args:
        requests (list): A list of 3 element tuples which are passed to the
            retrieve function of the CDS API client in order to retrieve
            the intended data.
        processing_class (`Worker` subclass): A subclass of `Worker` that
            defines a process method and takes and index (int) and a pipe
            (multiprocessing.connection.Connection) as constructor
            arguments. See `AveragingWorker` for a sample implementation.
        n_threads (int): The maximum number of data download threads to
            open at any one time. This corresponds to the number of open
            requests to the CDS API.
        soft_filesize_limit (float): If the cumulative size of downloaded
            and processed files on disk (see `delete_processed`) in GB
            exceeds this value, downloading of new files (ie. spawning of
            new threads) will cease until the aforementioned cumulative
            size drops below the threshold again. Exceeding the threshold
            does not terminate existing download threads or the processing
            worker, meaning that at most `n_threads - 1` downloads and
            processing of files in the worker queue could still occur
            before requests and processing are ceased entirely. If None, no
            limit is applied.
        timeout (float or str): Time-out after which the function
            terminates and ceases downloads as well as processing. If None, no
            limit is applied. If given as a float, the units are in
            seconds. A string may be given, in which case the units may be
            dictated. For example, '1s' would refer to one second, '1m' to
            one minute, '1h' to one hour, and '2d' to two days.
        delete_processed (bool): If True, remove downloaded files that have
            been successfully processed.
        overwrite (bool): If True, download files which have already been
            downloaded again. Note that only the time coordinate and
            variables are compared against for the existing files.

    Todo:
        Soft time-out which would not cause an abrupt exit of the program
        (using an Exception) but would allow a graceful exit.

        Check that downloaded files have correct grid (not relevant if grid
        is not specified and only the default is retrieved).

        Check that downloaded files have the correct units, which can be
        achieved by using wildfires.data.era5_tables.get_table_dict, as it
        returns a dictionary mapping long variable names to (amongst
        others) units.

    """
    start_time = time()
    if isinstance(timeout, str):
        timeout = str_to_seconds(timeout)
    requests = requests.copy()
    timeout_ramp = RampVar(2, 30, 10)
    threads = []
    remaining_files = []
    total_files = []
    processed_files = []
    issued_filesize_warning = False
    if delete_processed:
        raw_files = remaining_files
    else:
        raw_files = total_files

    worker_index = 0
    pipe_start, pipe_end = Pipe()
    processing_worker = processing_class(worker_index, pipe_end)
    processing_worker.start()
    worker_index += 1

    retrieve_queue = Queue()

    while requests or remaining_files or threads:
        if time() - start_time > timeout:
            raise RuntimeError("Timeout exceeded (timeout was {})."
                               .format(timeout))
        to_remove = []
        for thread in threads:
            if not thread.is_alive():
                logger.info('Worker {} has finished. Joining DownloadThread.'
                            .format(thread.id_index))
                to_remove.append(thread)
        for completed_thread in to_remove:
            completed_thread.join(1.)
            threads.remove(completed_thread)

        logger.info("Remaining files to process: {}."
                    .format(len(remaining_files)))
        logger.debug("Remaining files to process: {}.".format(remaining_files))
        logger.info("Number of remaining requests to process: {}."
                    .format(len(requests)))
        logger.info("Remaining active threads: {}.".format(len(threads)))

        new_threads = []
        while len(threads) < n_threads and requests:
            check_files = raw_files + processed_files
            filesize_sum = (sum([os.path.getsize(f) for f in check_files])
                            / 1000**3)
            if filesize_sum < soft_filesize_limit:
                if issued_filesize_warning:
                    issued_filesize_warning = False
                    logger.warning(
                        "Soft file size limit no longer exceeded. Requested "
                        "limit: {0.1e} GB. Observed: {:0.1e} GB. Active "
                        "threads: {}."
                        .format(soft_filesize_limit, filesize_sum,
                                len(threads)))
                new_request = requests.pop()
                new_filename = new_request[2]
                expected_output = (
                    0,
                    new_filename,
                    processing_worker.output_filename(new_filename)
                    )
                if not overwrite and os.path.isfile(expected_output[2]):
                    if processing_worker.check_output(
                            new_request, expected_output):
                        logger.info(
                            "'{}' already contains correct processed data. "
                            "Not downloading raw data."
                            .format(expected_output[2]))
                        continue
                if not overwrite and os.path.isfile(new_filename):
                    if (DownloadThread.retrieve_request(new_request)
                            == new_request):
                        logger.warning(
                            "'{}' already contains raw data for this "
                            "request. But no processed data has been found. "
                            "Sending request for processing now."
                            .format(new_filename))
                        # Taking a shortcut here - instead of spawning a
                        # new DownloadThread and downloading the file,
                        # emulate the behaviour of a DownloadThread by
                        # sending the request (for which there is already
                        # data, as required) to the queue normally used by
                        # the DownloadThread instances, thereby signalling
                        # that the file has been successfully 'downloaded',
                        # ie. it is available for processing.
                        retrieve_queue.put(new_request)
                        # It takes some time for the queue to register this.
                        # Allow up to 0.5 seconds for this to happen (just
                        # to be sure).
                        check_start = time()
                        while time() - check_start < 0.5:
                            if not retrieve_queue.empty():
                                break
                            sleep(0.01)
                        else:
                            assert not retrieve_queue.empty(), (
                                "The queue should not be empty as we just "
                                "called put.")
                        continue
                    else:
                        logger.warning(
                            "'{}' contains raw data for another request. "
                            "Deleting this file and retrieving new request."
                            .format(new_filename))
                        os.remove(new_filename)
                new_thread = DownloadThread(
                    worker_index, new_request, retrieve_queue)
                new_threads.append(new_thread)
                threads.append(new_thread)
                worker_index += 1
            else:
                logger.warning(
                    "Soft file size limit exceeded. Requested limit: {0.1e} "
                    "GB. Observed: {:0.1e} GB. Active threads: {}."
                    .format(soft_filesize_limit, filesize_sum, len(threads)))
                issued_filesize_warning = True

        logger.debug("Starting {} new threads.".format(len(new_threads)))
        for new_thread in new_threads:
            new_thread.start()

        logger.info("Active threads: {}.".format(len(threads)))

        queue_empty = retrieve_queue.empty()
        logger.debug("Retrieve queue is empty: {}.".format(queue_empty))
        if threads or not queue_empty:
            # Wait for (at least) one of the threads to successfully download
            # something.
            logger.debug("Waiting for a DownloadThread to finish.")
            retrieve_outputs = []
            get_timeout = timeout + ((start_time - time()) / 2)
            # This would only fail if the lines between here and the
            # previous timeout check involving time() took an exorbitantly
            # large time.
            assert get_timeout > 0, (
                "Time-out should be positive. Got {} instead."
                .format(get_timeout))
            try:
                retrieve_outputs.append(
                    retrieve_queue.get(timeout=get_timeout))
            except Empty:
                logger.exception("No data downloaded within {:0.1e} s."
                                 .format(get_timeout))
            # If there is something else in the queue, retrieve this until
            # nothing is left.
            while not retrieve_queue.empty():
                retrieve_outputs.append(retrieve_queue.get())

            logger.debug("DownloadThread output: {}.".format(retrieve_outputs))
            # Handle the output.
            for retrieve_output in retrieve_outputs:
                if (hasattr(retrieve_output, '__len__')
                        and len(retrieve_output) > 1
                        and isinstance(retrieve_output[1], Exception)):
                    # Re-raise exception here complete with traceback, to
                    # use logging exception convenience function.
                    try:
                        raise retrieve_output[1].with_traceback(
                                retrieve_output[2])
                    except Exception:
                        logger.exception("Exception while downloading data.")
                        continue
                # If it is a filename and not an exception, add this to the
                # queue for the processing worker.
                logger.debug("Sending filename to worker: {}."
                             .format(retrieve_output[2]))
                pipe_start.send(retrieve_output)
                remaining_files.append(retrieve_output[2])
                total_files.append(retrieve_output[2])

        if not threads and not requests and not remaining_files:
            continue
        elif threads:
            worker_timeout = 0
            timeout_ramp.reset()
        else:
            worker_timeout = timeout_ramp.value

        logger.debug("Polling processing worker pipe with timeout={:0.1f}."
                     .format(worker_timeout))
        # The call to poll will block for `worker_timeout` seconds.
        while pipe_start.poll(timeout=worker_timeout):
            output = pipe_start.recv()
            # The output may contain sys.exc_info() in case of an Exception.
            # sys.exc_info(): [0] contains the exception class, [1] the
            # instance (ie. that's what would be used for isinstance()
            # checking!) and [2] contains the traceback object, to be used
            # like: raise
            # sys.exc_info()[1].with_traceback(sys.exc_info()[2])
            if (hasattr(output, '__len__')
                    and len(output) > 1
                    and isinstance(output[1], Exception)):
                # Re-raise exception here complete with traceback, to
                # use logging exception convenience function.
                try:
                    raise output[1].with_traceback(output[2])
                except Exception:
                    logger.exception("Exception while processing data.")
            # The first entry of the output represents a status code.
            elif output[0] == 0:
                logger.info("Processed file '{}' successfully."
                            .format(output[1]))
            elif output[0] == 1:
                logger.error("Error while processing {}"
                             .format(output[1]))
            elif output[0] == 2:
                logger.error(
                    "Processing output for {} did not match expected output."
                    .format(output[1]))
            else:
                raise ValueError("Unknown output format:{}.".format(output))

            remaining_files.remove(output[1])
            processed_files.append(output[2])
            if delete_processed:
                logger.info("Deleting file '{}' as it has been processed."
                            .format(output[1]))
                os.remove(output[1])
            if not remaining_files:
                # Everything should have been handled so we can exit.
                break
    else:
        logger.info("No remaining requests, files or threads.")

    # After everything is done, terminate the processing process (by force
    # if it exceeds the time-out).
    logger.info("Terminating AveragingWorker.")
    pipe_start.send("STOP_WORKER")
    processing_worker.join(timeout=20)
    processing_worker.terminate()


if __name__ == '__main__':
    logging.config.dictConfig(LOGGING)

    requests = retrieve_hourly(
            variable=['2t', '10u', '10v'],
            start=PartialDateTime(2005, 1, 1),
            end=PartialDateTime(2005, 2, 1))
    retrieval_processing(
            requests, n_threads=1, delete_processed=True, overwrite=False,
            soft_filesize_limit=30)
