#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.expanduser(os.path.join("~", "Documents", "wildfires", "data")))
# print('path')
# from pprint import pprint
# pprint(sys.path)
# pprint(os.listdir(sys.path[-1]))
import logging

logging.basicConfig(level=logging.DEBUG)
from datasets import CHELSA

# 1680 files

index = int(os.environ["PBS_ARRAY_INDEX"])

print("Value of ARRAY_ID:{:}".format(index))
CHELSA(slice(index, index + 1))
