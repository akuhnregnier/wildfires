#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import logging.config
import os
import sys

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    sys.path.append(os.path.expanduser(os.path.join("~", "Documents", "wildfires")))
    from wildfires.data.datasets import CHELSA
    from wildfires.logging_config import LOGGING

    logging.config.dictConfig(LOGGING)

    # 1680 files

    index = int(os.environ["PBS_ARRAY_INDEX"])

    logger.info("Value of ARRAY_ID:{:}".format(index))
    CHELSA(slice(index, index + 1))
