#!/usr/bin/env python3
import sys
import os
from datasets import Copernicus_SWI, LOGGING
import logging
import logging.config
logger = logging.getLogger(__name__)
logging.config.dictConfig(LOGGING)

index = int(os.environ['PBS_ARRAY_INDEX'])

logger.info("Value of ARRAY_ID:{:}".format(index))
a = Copernicus_SWI(slice(index, index+1))
