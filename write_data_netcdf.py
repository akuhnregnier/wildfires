#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import calendar
import glob
import os
import shutil
from textwrap import dedent


def convert_to_netcdf(d):
    if not os.path.isdir(d):
        raise ValueError("Directory {:} does not exist!".format(d))
    time_str = os.path.split(d)[-1]
    if not time_str:
        time_str = os.path.split(os.path.dirname(d))[-1]
    year = int(time_str[:4])
    month = int(time_str[4:])

    assert 2000 <= year <= 2023
    assert 1 <= month <= 12

    print("year:", year)
    print("month:", month)

    # Construct a '.gs' file in each directory which will instruct the
    # grads program to carry out the necessary steps for each directory.
    # The program also needs a '.ctl' file.

    # Extract the number of days of data in the directory

    min_day = int(
        os.path.split((sorted(glob.glob(os.path.join(d, "*.dat")))[0]))[1][18:20]
    )
    max_day = int(
        os.path.split((sorted(glob.glob(os.path.join(d, "*.dat")))[-1]))[1][18:20]
    )

    gs_file = (
        dedent(
            """
        'open GSMaP_NRT_{:}.daily_G.00Z-23Z.ctl'
        'set t {:} {:}'
        'define precip = precip'
        'set sdfwrite GSMaP_G_{:}_0_25d_daily_00Z_23Z.nc'
        'sdfwrite precip'
        """.format(
                time_str, min_day, max_day, time_str
            )
        ).strip()
        + "\n"
    )

    print("gs file:", gs_file)
    print("writing gs file to:", d)

    with open(os.path.join(d, "convert_to_netcdf.gs"), "w") as f:
        f.write(gs_file)

    ctl_file = (
        dedent(
            """
        DSET ^gsmap_gauge.%y4%m2%d2.0.25d.daily.00Z-23Z.dat
        TITLE GSMaP_GAUGE 0.25deg Daily (00:00Z-23:59Z)
        UNDEF -999.9
        OPTIONS YREV LITTLE_ENDIAN TEMPLATE
        XDEF 1440 LINEAR    0.125 0.25
        YDEF 480 LINEAR  -59.875 0.25
        zdef 1 levels 1013
        tdef 10950 linear 00:00z{:}{:}{:} 1dy
        VARS 1
        precip          0 99 daily averaged precip(mm/hr)
        ENDVARS
        """.format(
                min_day, calendar.month_abbr[month].lower(), year
            )
        ).strip()
        + "\n"
    )
    # Yes, the month abbreviations above are 1-indexed, as months start
    # from 1 not 0...

    print("ctl file:", ctl_file)
    print("writing ctl file to:", d)

    with open(
        os.path.join(d, "GSMaP_NRT_{:}.daily_G.00Z-23Z.ctl".format(time_str)), "w"
    ) as f:
        f.write(ctl_file)

    # Run grads in the directory using the files above.
    cmd = 'cd {:} && grads -lbxc "convert_to_netcdf.gs"'.format(d)
    print("executing:{:}".format(cmd))
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=dedent(
            """
            Convert daily data files in given directory to a NetCDF file
            containing aggregated monthly data in the same directory using
            the grads program.

            Can be used for many directories sequentially by using command
            line utilities, such as:

            find . -type d -exec ./write_data_netcdf.py -d {} \;
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-d", help=("""The directory within which the daily files are stored.""")
    )

    args = parser.parse_args()

    d = args.d
    files = os.listdir(d)
    dat_file = [".dat" in s for s in files]
    if True in dat_file:
        convert_to_netcdf(d)
    else:
        print("Skipping, as no .dat files found in directory:'{:}'".format(d))
