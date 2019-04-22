#!/usr/bin/env pytyhon3
# -*- coding: utf-8 -*-
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from pprint import pprint
import re

from wildfires.data.datasets import DATA_DIR


if __name__ == "__main__":
    files = os.listdir(os.path.join(DATA_DIR, "Copernicus_SWI", "monthly"))
    pattern = re.compile(r"(\d{4})(\d{2})(\d{2})")
    datetimes = sorted([datetime(*map(int, pattern.search(f).groups())) for f in files])

    start = datetime(2007, 1, 1)
    end_year_month = datetime(2019, 3, 1)

    reference_datetimes = []
    dt = start
    while dt <= end_year_month:
        reference_datetimes.append(dt)
        dt += relativedelta(months=+1)

    missing_indices = []

    i = 0
    i_2 = 0
    while i < len(reference_datetimes):
        if reference_datetimes[i] != datetimes[i_2]:
            print("Missing:", reference_datetimes[i])
            missing_indices.append(i)
            i_2 -= 1

        i += 1
        i_2 += 1

    print("Missing:")
    pprint(missing_indices)
