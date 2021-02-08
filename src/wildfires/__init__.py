# -*- coding: utf-8 -*-
"""Utilities for the analysis of wildfires."""
from numba import set_num_threads

from ._version import version as __version__
from .qstat import get_ncpus

del _version

set_num_threads(get_ncpus())
