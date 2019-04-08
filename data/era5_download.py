#!/usr/bin/env python3
import logging
import logging.config
from multiprocessing import Process, Pipe, Queue
import os
import sys
from threading import Thread


import cdsapi
from datetime import datetime
from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime
import numpy as np

from wildfires.data.datasets import DATA_DIR
from wildfires.logging_config import LOGGING


logger = logging.getLogger(__name__)


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
    else:
        raise TypeError(type_error_msg)


def retrieve_hourly(variable='2m_temperature', levels='sfc', hours=None,
                    start=PartialDateTime(2000, 1, 1),
                    end=PartialDateTime(2000, 2, 1), target_dir=DATA_DIR):
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
        target (str): Directory path where the output files will be stored.

    Returns:
        list: list of output filenames. There will be one output filename
            per month containing all of the requested variables, named like
            era5_hourly_reanalysis_{year}_{month}.nc.

    Note:
        Reanalysis data is retrieved.

        Variables may have different names depending on whether the 'sfc'
        level or a pressure level is requested.

        Time information (ie. hours, minutes, etc...) in the start and end
        arguments will be ignored.

    """
    # TODO: Won't need this anymore when Threads are implemented.
    client = cdsapi.Client()

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
    logger.debug('Requested hours:{}.'.format(hours))

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
    logger.debug('Requested levels:{}.'.format(levels))

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

    for request_date in monthly_dates.values():
        logger.debug('Requesting dates:{}.'.format(request_date))
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
        client.retrieve(*request)
        logger.info("Finished download to:'{}'.".format(target_file))


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
        list(set([year // 10 for year in range(start.year, end.year + 1)])))
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

        client.retrieve("reanalysis-era5-complete",
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
                'decade': str(decade), 'grid': '0.25/0.25', # Optional. Subset (clip) to an area. Specify as N/W/S/E in
                # Geographic lat/long degrees. Southern latitudes and western
                # longitudes must be given as negative numbers. Requires "grid"
                # to be set to a regular grid, e.g. "0.25/0.25".
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

    """

    def __init__(self, initial_value, max_value, steps):
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


class LoggingMixin:
    """Relies on child classes defining a 'logger' and 'id_index'
    attribute to emit logging messages.

    """

    def log(self, msg, level=logging.DEBUG):
        """Emit logging messages at the specified level."""
        msg = '{:03d} | {}'.format(self.id_index, msg)
        self.logger.log(level, msg)


import urllib3
import shutil
from time import sleep

def retrieve_func(index):
    logger.info("DOWNLOADING file.")
    url = 'http://ipv4.download.thinkbroadband.com:8080/200MB.zip'
    c = urllib3.PoolManager(headers={'user-agent': 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) ..'})
    with c.request('GET', url, preload_content=False) as resp, open('/tmp/{}_download'.format(index), 'wb') as out_file:
        shutil.copyfileobj(resp, out_file)
    resp.release_conn()
    logger.info("FINISHED downloading.")


class DownloadThread(Thread, LoggingMixin):

    def __init__(self, id_index, request, queue, logger):
        super().__init__()
        self.id_index = id_index
        self.request = request
        self.queue = queue
        self.logger = logger
        # TODO: Restore this!
        # self.client = cdsapi.Client()

        def retrieve(*args):
            self.log("Retrieving request: {}".format(args))
            retrieve_func(self.id_index)

        self.client = lambda: None
        self.client.retrieve = retrieve

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
            # sys.exc_info(): [0] contains the exception class, [1] the
            # instance (ie. that's what would be used for isinstance()
            # checking!) and [2] contains the traceback object, to be used
            # like: raise
            # sys.exc_info()[1].with_traceback(sys.exc_info()[2])
            self.queue.put(sys.exc_info())
        finally:
            self.log("Exiting.")
            pass


class AveragingWorker(Process, LoggingMixin):

    def __init__(self, id_index, pipe, logger):
        super().__init__()
        self.id_index = id_index
        self.pipe = pipe
        self.logger = logger

        self.log("Initialised AveragingWorker with id_index={}."
                 .format(self.id_index))

    def process(self, filename):
        # TODO
        self.log('Processing: {}.'.format(filename))
        # If everything went well.
        return (0, filename)

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
                status = self.process(file_to_process)
                self.log("Finished processing '{}' with status {}."
                         .format(file_to_process, status))
                self.pipe.send(status)
                self.log("Sent status {}.".format(status))
        except Exception:
            # sys.exc_info(): [0] contains the exception class, [1] the
            # instance (ie. that's what would be used for isinstance()
            # checking!) and [2] contains the traceback object, to be used
            # like: raise
            # sys.exc_info()[1].with_traceback(sys.exc_info()[2])
            self.pipe.send(sys.exc_info())
        finally:
            self.log("Exiting.")
            pass


def spawn_retrieve_processing(requests, n_threads=1):
    # TODO: Update this to match the actual implementation!!
    """Start retrieving and processing data asynchronously.

    Calling process spawns one non-blocking process just for the calculation of
    the mean values for one pre-downloaded file.

    After this processing process has finished, it will be noted that this file
    has been successfully processed (barring any errors that were received,
    which should be logged!) and removed from the list of files that are
    available for processing.

    A number of threads will be started in order to download data concurrently.
    If one Thread finishes downloading data, it will receive a new request to
    execute, and so on for the other threads, until all the requests have been
    handled.

    When one of these threads has finished downloading data (again, check for
    errors here!) signal to the main process that a file has been downloaded
    successfully. The main process will then add the downloaded file's name
    to the list of names that have yet to be processed, and start giving these
    names to the dedicated processing process.

    Args:
        requests (list): A list of 3 element tuples which are passed to the
            retrieve function of the CDS API client in order to retrieve
            the intended data.

    """
    requests = requests.copy()
    worker_index = 0
    pipe_start, pipe_end = Pipe()
    averaging_worker = AveragingWorker(worker_index, pipe_end, logger)
    averaging_worker.start()
    worker_index += 1

    timeout_ramp = RampVar(0.5, 10, 15)
    retrieve_queue = Queue()

    threads = []
    remaining_files = []
    while requests or remaining_files or threads:
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
            new_thread = DownloadThread(
                worker_index, requests.pop(), retrieve_queue, logger)
            new_threads.append(new_thread)
            threads.append(new_thread)
            worker_index += 1

        for new_thread in new_threads:
            new_thread.start()

        if threads:
            # Wait for (at least) one of the threads to successfully download
            # something.
            logger.debug("Waiting for a DownloadThread to finish.")
            retrieve_outputs = []
            retrieve_outputs.append(retrieve_queue.get())
            # If there is something else in the queue, retrieve this until
            # nothing is left.
            while not retrieve_queue.empty():
                retrieve_outputs.append(retrieve_queue.get())

            logger.debug("DownloadThread output: {}.".format(retrieve_outputs))
            # Handle the output.
            for retrieve_output in retrieve_outputs:
                # TODO: Handle this?
                if isinstance(retrieve_output[1], Exception):
                    raise retrieve_output[1].with_traceback(retrieve_output[2])
                # If it is a filename and not an exception, add this to the
                # queue for the processing worker.
                logger.debug("Sending filename to worker: {}."
                             .format(retrieve_output))
                pipe_start.send(retrieve_output)
                remaining_files.append(retrieve_output)

        if not threads and not requests and not remaining_files:
            logger.info("No remaining requests, files or threads.")
            break
        elif threads:
            worker_timeout = 0
        else:
            worker_timeout = timeout_ramp.value

        # from time import sleep
        # sleep(0.2)

        logger.debug("Polling processing worker pipe with timeout={:0.1f}."
                     .format(worker_timeout))
        while pipe_start.poll(timeout=worker_timeout):
            output = pipe_start.recv()
            # TODO: Handle this?
            if isinstance(output[1], Exception):
                raise output[1].with_traceback(output[2])
            # The output represents a status code.
            if output[0] == 0:
                logger.info("Processed file '{}' successfully. Removing it "
                            "from list of files to process."
                            .format(output[1]))
                remaining_files.remove(output[1])
                # Everything should have been handled if this is true.
                if not remaining_files:
                    break
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
    averaging_worker.join(timeout=2)
    averaging_worker.terminate()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    logging.config.dictConfig(LOGGING)

    # from logging_tree import printout
    # printout()

    # retrieve_monthly(start=PartialDateTime(2000, 1),
    #                  end=PartialDateTime(2000, 1))

    # retrieve_hourly(
    #         variable='temperature',
    #         start=PartialDateTime(2000, 1, 1),
    #         end=PartialDateTime(2000, 1, 3),
    #         levels='100',
    #         target_dir='/tmp')

    spawn_retrieve_processing([
        ('dataset-name', {'a': 'test'}, '/tmp/filename.smth'),
        ('dataset-name', {'a': 'test'}, '/tmp/filename.smth')
        ],
        n_threads=2)

