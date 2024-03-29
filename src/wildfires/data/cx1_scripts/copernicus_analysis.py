#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import logging.config
import os
import sys

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    sys.path.append(os.path.expanduser(os.path.join("~", "Documents", "wildfires")))
    from wildfires.data.datasets import LOGGING, Copernicus_SWI

    logger = logging.getLogger(__name__)
    logging.config.dictConfig(LOGGING)

    index = int(os.environ["PBS_ARRAY_INDEX"])

    to_calculate = [
        12,
        32,
        37,
        38,
        39,
        57,
        61,
        63,
        66,
        67,
        74,
        79,
        80,
        82,
        83,
        87,
        89,
        91,
        93,
        103,
        104,
        110,
        117,
        126,
        128,
        129,
        130,
        131,
        132,
        135,
        136,
        137,
    ]

    index = to_calculate[index]

    logger.info("Value of ARRAY_ID:{:}".format(index))
    a = Copernicus_SWI(slice(index, index + 1))
