#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
from wildfires.data.datasets import Copernicus_SWI, LOGGING
import logging
import logging.config
logger = logging.getLogger(__name__)
logging.config.dictConfig(LOGGING)

index = int(os.environ['PBS_ARRAY_INDEX'])

logger.info("Value of ARRAY_ID:{:}".format(index))
a = Copernicus_SWI(slice(index, index+1))
