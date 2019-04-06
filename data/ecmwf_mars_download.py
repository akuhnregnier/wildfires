#!/usr/bin/env python3
import logging

import cdsapi
from dateutil.relativedelta import relativedelta
from iris.time import PartialDateTime
import numpy as np


logger = logging.getLogger(__name__)


def retrieve_monthly_era5(param='167.128', start=PartialDateTime(2000, 1),
                          end=PartialDateTime(2010, 1)):
    """Retrieve monthly ERA5 data for the chosen parameter.

    Args:
        param (str): Variable of interest: eg. param='167.128' refers to
            the 2m surface (sfc) level temperature variable. Multiple
            variables are also possible: eg. param='167.128/207.128' would
            retrieve 2m temperature and 10m wind speed.
        start (datetime): Initial datetime. This is inclusive.
        end (datetime): Final datetime. This is inclusive.

    Note:
        Daily information present in the supplied dates is simply discarded.

    The output data are organised as one file per decade:
     - ...
     - 'era5_moda_{param}_1990'
     - 'era5_moda_{param}_2000'
     - ...

    """
    c = cdsapi.Client()
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
        set([year // 10 for year in range(start.year, end.year + 1)]))
        * 10))

    for decade in decades:
        requested_dates = []
        # The decade is the first year, so go from there until the final
        # year, up to 9 times (otherwise the next decade would be reached).
        for year in range(decade, min((end.year, decade + 10))):
            for month in range(1, end.month if year == end.year else 13):
                requested_dates.append('{:>04d}{:>02d}01'.format(year, month))

        date_request_string = '/'.join(requested_dates)
        decade_target_file = 'era5_moda_{}_{}.nc'.format(param, decade)

        c.retrieve("reanalysis-era5-complete",
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
                'decade': str(decade),
                'grid': '0.25/0.25',
                # Optional. Subset (clip) to an area. Specify as N/W/S/E in
                # Geographic lat/long degrees. Southern latitudes and western
                # longitudes must be given as negative numbers. Requires "grid"
                # to be set to a regular grid, e.g. "0.25/0.25".
                # 'area': '89.75/-179.75/-89.75/179.75',
                'format': 'netcdf'
            },
            decade_target_file)
        logger.debug('Finished download for decade:{} to:{}'
                     .format(decade, decade_target_file))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    """
    Sample retrieval below.
    """
    c = cdsapi.Client()
    # target_file = "/tmp/era5_download_test_area.nc"
    target_file = "/tmp/test.nc"
    # "reanalysis-era5-complete" is not listed publicly, only in
    # https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
    # It should contain monthly means?
    c.retrieve("reanalysis-era5-complete",
        {
            'class': 'ea',
            'expver': '1',
            'stream': 'moda',
            'type': 'an',
            # 128 is table 2 version
            # https://confluence.ecmwf.int/display/UDOC/Identification+keywords
            'param': '167.128',
            'levtype': 'sfc',
            'date': '2018-01-01',
            'decade': '2010',
            'grid': '0.25/0.25',
            # Optional. Subset (clip) to an area. Specify as N/W/S/E in
            # Geographic lat/long degrees. Southern latitudes and western
            # longitudes must be given as negative numbers. Requires "grid"
            # to be set to a regular grid, e.g. "0.25/0.25".
            # 'area': '89.75/-179.75/-89.75/179.75',
            'format': 'netcdf'
        },
        target_file)
    logger.debug('Finished download!')
