#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from wildfires.data.datasets import DATA_DIR


if __name__ == "__main__":
    combinations = []

    for year in range(2000, 2018):
        for estimate_type in ["upper", "baseline", "lower"]:
            for dataset in ["lu", "pop"]:
                combinations.append((year, estimate_type, dataset))

    for year, estimate_type, dataset in tqdm(combinations):
        # url = 'ftp://ftp.pbl.nl/hyde/hyde3.2/upper/zip/2004AD_pop.zip'
        url = "ftp://ftp.pbl.nl/hyde/hyde3.2/{:}/zip/{:04d}AD_{:}.zip".format(
            estimate_type, year, dataset
        )
        save_as = os.path.join(
            DATA_DIR, "HYDE", estimate_type, "{:04d}AD_{:}.zip".format(year, dataset)
        )
        if not os.path.isdir(os.path.dirname(save_as)):
            os.makedirs(os.path.dirname(save_as))
        command = "curl --connect-timeout 20 -L -o {:} {:}".format(save_as, url)
        os.system(command)
