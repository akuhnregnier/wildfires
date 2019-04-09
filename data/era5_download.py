#!/usr/bin/env python3
"""Tools for downloading ERA5 data using the CDS API.

"""
from datetime import datetime
import logging
import logging.config
from multiprocessing import Process, Pipe, Queue
import os
import sys
from threading import Thread
from time import time
from queue import Empty

import cdsapi
from dateutil.relativedelta import relativedelta
import iris
from iris.time import PartialDateTime
import numpy as np

from wildfires.data.datasets import DATA_DIR
from wildfires.logging_config import LOGGING


logger = logging.getLogger(__name__)


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
        client = cdsapi.Client()
    else:
        client = None

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
    client = cdsapi.Client()
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


class LoggingMixin:
    """Relies on child classes defining a 'logger' and 'id_index'
    attribute to emit logging messages.

    """

    def log(self, msg, level=logging.DEBUG):
        """Emit logging messages at the specified level."""
        msg = '{:03d} | {}'.format(self.id_index, msg)
        self.logger.log(level, msg)


class DownloadThread(Thread, LoggingMixin):
    """Retrieve data using the CDS API."""

    def __init__(self, id_index, request, queue, logger):
        super().__init__()
        self.id_index = id_index
        self.request = request
        self.queue = queue
        self.logger = logger
        self.client = cdsapi.Client()
        self.log("Initialised DownloadThread with id_index={}."
                 .format(self.id_index))

    def run(self):
        try:
            self.log('Requesting: {}'.format(self.request))
            self.client.retrieve(*self.request)
            self.log('Completed request.')
            filename = self.request[2]
            if not os.path.isfile(filename):
                raise RuntimeError(
                    "Filename '{}' not found despite request "
                    "{} having being issued.".format(
                        filename, self.request))
            self.queue.put(filename)
        except Exception:
            self.queue.put(sys.exc_info())
        finally:
            self.log("Exiting.")


class AveragingWorker(Process, LoggingMixin):
    """Compute monthly averages using filenames passed in via a pipe."""

    def __init__(self, id_index, pipe, logger):
        super().__init__()
        self.id_index = id_index
        self.pipe = pipe
        self.logger = logger

        self.log("Initialised AveragingWorker with id_index={}."
                 .format(self.id_index))

    def process(self, filename):
        """Performs monthly averaging on the data in the given file.

        Args:
            filename (str): Specifies the file holding the data to be averaged.

        Returns:
            int: 0 for successful computation of the average. 1 is returned
                if an error is encountered. Note that exceptions are merely
                logged and not raised.
            str: The original filename.
            str or None: The output filename (if successful) or None.

        """
        self.log('Processing: {}.'.format(filename))
        try:
            cubes = iris.load(filename)
            cubes = iris.cube.CubeList(
                [cube.collapsed('time', iris.analysis.MEAN) for cube in cubes])
            # Realise data so we don't end up storing dask operations
            # instead of numpy arrays.
            [cube.data for cube in cubes]
            save_name = filename.split('.nc')[0] + '_monthly_mean.nc'
            iris.save(cubes, save_name, zlib=False)
            # If everything went well.
            return (0, filename, save_name)
        except Exception:
            logger.exception("Error during processing of '{}'."
                             .format(filename))
            return (1, filename, None)

    def run(self):
        try:
            self.log("Started listening for filenames to process.")
            while True:
                file_to_process = self.pipe.recv()
                if file_to_process == "STOP_WORKER":
                    logger.debug(
                        "STOP_WORKER received, breaking out of loop.")
                    break
                self.log("Received file: '{}'. Starting processing."
                         .format(file_to_process))
                output = self.process(file_to_process)
                self.log("Finished processing '{}' with status {}."
                         .format(file_to_process, output))
                self.pipe.send(output)
                self.log("Sent status {}.".format(output))
        except Exception:
            self.pipe.send(sys.exc_info())
        finally:
            self.log("Exiting.")


def retrieval_processing(requests, n_threads=4, soft_filesize_limit=50,
                         timeout='3d', delete_processed=True, overwrite=False):
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

    Args:
        requests (list): A list of 3 element tuples which are passed to the
            retrieve function of the CDS API client in order to retrieve
            the intended data.
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
            downloaded again. Note that no checks are made for broken
            files due to interrupted downloads or some similar error (Todo).

    Todo:
        Soft time-out which does not cause an abrupt exit of the program
        (using an Exception) but allows the threads/process to exit
        gracefully.

        Check that downloaded files (or pre-existing downloaded files in
        case of overwrite=False) are intact and match the request contents.

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
    averaging_worker = AveragingWorker(worker_index, pipe_end, logger)
    averaging_worker.start()
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

        logger.debug("Current number of threads: {}.".format(len(threads)))
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
                if os.path.isfile(new_filename) and not overwrite:
                    logger.info("'{}' already exists. Not downloading again."
                                .format(new_filename))
                    continue
                new_thread = DownloadThread(
                    worker_index, new_request, retrieve_queue, logger)
                new_threads.append(new_thread)
                threads.append(new_thread)
                worker_index += 1
            else:
                logger.warning(
                    "Soft file size limit exceeded. Requested limit: {0.1e} "
                    "GB. Observed: {:0.1e} GB. Active threads: {}."
                    .format(soft_filesize_limit, filesize_sum, len(threads)))
                issued_filesize_warning = True

        for new_thread in new_threads:
            new_thread.start()

        if threads:
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
                if isinstance(retrieve_output[1], Exception):
                    # Re-raise exception here complete with traceback, to
                    # use logging exception convenience function.
                    try:
                        raise retrieve_output[1].with_traceback(
                                retrieve_output[2])
                    except Exception:
                        logger.exception("Exception while downloading data.")
                # If it is a filename and not an exception, add this to the
                # queue for the processing worker.
                logger.debug("Sending filename to worker: {}."
                             .format(retrieve_output))
                pipe_start.send(retrieve_output)
                remaining_files.append(retrieve_output)
                total_files.append(retrieve_output)

        if not threads and not requests and not remaining_files:
            logger.info("No remaining requests, files or threads.")
            break
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
            if isinstance(output[1], Exception):
                # Re-raise exception here complete with traceback, to
                # use logging exception convenience function.
                try:
                    raise output[1].with_traceback(output[2])
                except Exception:
                    logger.exception("Exception while processing data.")
            # The first entry of the output represents a status code.
            if output[0] == 0:
                logger.info("Processed file '{}' successfully. Removing it "
                            "from list of files to process."
                            .format(output[1]))
                remaining_files.remove(output[1])
                processed_files.append(output[2])
                if delete_processed:
                    logger.info("Deleting file '{}' as it has been "
                                "processed successfully"
                                .format(output[1]))
                    os.remove(output[1])
                if not remaining_files:
                    # Everything should have been handled so we can exit.
                    break
            elif output[0] == 1:
                logger.warning("Error during process of {}"
                               .format(output[1]))
            else:
                raise NotImplementedError("Other codes not handled yet!")

        logger.info("Remaining files to process: {}.".format(remaining_files))
        logger.info("Threads: {}.".format(threads))
        logger.info("Number of remaining requests to process: {}."
                    .format(requests))

    # After everything is done, terminate the processing process (by force
    # if it exceeds the time-out).
    logger.info("Terminating AveragingWorker.")
    pipe_start.send("STOP_WORKER")
    averaging_worker.join(timeout=20)
    averaging_worker.terminate()


if __name__ == '__main__':
    logging.config.dictConfig(LOGGING)

    from logging_tree import printout
    printout()

    # requests = retrieve_hourly(
    #         start=PartialDateTime(2001, 1, 1),
    #         end=PartialDateTime(2001, 2, 1))
    # retrieval_processing(requests, n_threads=1)

    print(logger.handlers)
    logger.info("Test")
    logger.debug('Test2')

    class A(LoggingMixin):
        def __init__(self, logger):
            super().__init__()
            self.logger = logger

        def testing(self, msg):
            self.log(msg)

    a = A()
    a.testing('mixing')
