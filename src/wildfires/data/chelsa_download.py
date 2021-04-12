#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# URLs e.g.
https://www.wsl.ch/lud/chelsa/data/timeseries/prec/CHELSA_prec_1979_01_V1.2.1.tif
https://www.wsl.ch/lud/chelsa/data/timeseries/prec/CHELSA_prec_2013_12_V1.2.1.tif

https://www.wsl.ch/lud/chelsa/data/timeseries/tmean/CHELSA_tmean_1979_01_V1.2.1.tif
https://www.wsl.ch/lud/chelsa/data/timeseries/tmean/CHELSA_tmean_1979_02_V1.2.1.tif

"""
import os

from tqdm import tqdm

from wildfires.configuration import DATA_DIR


def download(dataset, year, month, ignore_existing=True, size_threshold=8.8e7):
    url = (
        "https://www.wsl.ch/lud/chelsa/data/timeseries/{:}/"
        "CHELSA_{:}_{:}_{:02d}_V1.2.1.tif"
    ).format(dataset, dataset, year, month)

    save_as = url[23:]
    save_as = save_as[:6].upper() + save_as[6:]
    save_as = os.path.join(DATA_DIR, save_as)
    if not os.path.isdir(os.path.dirname(save_as)):
        os.makedirs(os.path.dirname(save_as))
    if (
        os.path.isfile(save_as)
        and ignore_existing
        and os.path.getsize(save_as) > size_threshold
    ):
        print(
            "File exists and is large enough, not " "downloading:'{:}'".format(save_as)
        )
    else:
        command = "curl --connect-timeout 20 -L -o {:} {:}".format(save_as, url)
        print(command + "\n")
        os.system(command)


if __name__ == "__main__":
    datasets = ["prec", "tmax", "tmean", "tmin"]

    combinations = []

    for year in range(1979, 2014):
        for month in range(1, 13):
            for dataset in datasets:
                combinations.append((dataset, year, month))

    for i in tqdm(range(len(combinations))):
        download(*combinations[i])
