# -*- coding: utf-8 -*-
"""Collection of qstat related code used throughout the project.

"""
import json
import logging
import os
from subprocess import DEVNULL, check_output

logger = logging.getLogger(__name__)


def get_qstat_json():
    """Retrieve the json representation of the qstat output.

    Raises:
        FileNotFoundError: If the command is not run on the hpc.

    """
    raw_output = check_output(("qstat", "-f", "-F", "json"), stderr=DEVNULL).decode()
    # Filter out invalid json (unescaped double quotes).
    filtered_lines = [line for line in raw_output.split("\n") if '"""' not in line]
    filtered_output = "\n".join(filtered_lines)
    return json.loads(filtered_output)


def get_ncpus(default=1):
    ncpus = os.environ.get("NCPUS")
    if ncpus:
        logger.info("Read ncpus: {} from NCPUS environment variable.".format(ncpus))
        return int(ncpus)
    logger.warning(
        "Could not read NCPUS environment variable. Using default: {}.".format(default)
    )
    return default
