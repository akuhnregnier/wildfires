# -*- coding: utf-8 -*-
import os

DATA_DIR = os.path.join(os.path.expanduser("~"), "FIREDATA")

# Above this mm/h threshold, a day is a 'wet day'.
# 0.1 mm per day, from Harris et al. 2014, as used in Forkel et al. 2018.
MM_PER_HR_THRES = 0.1 / 24
M_PER_HR_THRES = MM_PER_HR_THRES / 1000


def data_is_available():
    """Check if DATA_DIR exists.

    Returns:
        bool: True if the data directory exists.

    """
    return os.path.exists(DATA_DIR)
