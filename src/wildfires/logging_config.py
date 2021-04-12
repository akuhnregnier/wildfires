#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging.config
import os
import socket
from copy import deepcopy
from pathlib import PurePath

log_dir = os.path.expanduser(os.path.join("~", "Documents", "wildfire_logs"))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": (
                "%(asctime)s:%(levelname)-8s:%(name)-20s:%(filename)-25s:%(lineno)-5s"
                ":%(funcName)-30s:%(message)s"
            )
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "debug_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 7,
            "level": "DEBUG",
            "formatter": "default",
            "filename": os.path.join(log_dir, "wildfires_debug.log"),
        },
        "info_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 7,
            "level": "INFO",
            "formatter": "default",
            "filename": os.path.join(log_dir, "wildfires_info.log"),
        },
        "warning_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 7,
            "level": "WARNING",
            "formatter": "default",
            "filename": os.path.join(log_dir, "wildfires_warning.log"),
        },
        "root_debug_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 7,
            "level": "DEBUG",
            "formatter": "default",
            "filename": os.path.join(log_dir, "root_debug.log"),
        },
        "root_info_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 7,
            "level": "INFO",
            "formatter": "default",
            "filename": os.path.join(log_dir, "root_info.log"),
        },
        "root_warning_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 7,
            "level": "WARNING",
            "formatter": "default",
            "filename": os.path.join(log_dir, "root_warning.log"),
        },
    },
    "loggers": {
        "wildfires": {
            "level": "DEBUG",
            "handlers": ["console", "debug_file", "info_file", "warning_file"],
        },
        "": {
            "level": "DEBUG",
            "handlers": ["root_debug_file", "root_info_file", "root_warning_file"],
        },
    },
}

LOGGING["loggers"]["__main__"] = LOGGING["loggers"]["wildfires"]
# A copy of the usual configuration with a higher threshold for console output.
JUPYTER_LOGGING = deepcopy(LOGGING)
JUPYTER_LOGGING["handlers"]["console"]["level"] = "WARNING"


def enable_logging(mode="normal", level=None, pbs=False):
    """Configure logging in a standardised manner.

    Args:
        mode (str): Which configuration to use. Possible values are "normal" or
            "jupyter".
        level (str or logging level): If given, alter the console logger level. If a
            string is given, `level.upper()` will be used to retrieve the logging
            level.
        pbs (bool): Place logging files in the 'pbs' sub-directory. All files will
            additionally include the PBS job id and machine hostname.

    """
    if mode == "normal":
        config_dict = LOGGING.copy()
    elif mode == "jupyter":
        config_dict = JUPYTER_LOGGING.copy()
    else:
        raise ValueError(f"Unknown mode '{mode}'.")
    if level is not None:
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
        config_dict["handlers"]["console"]["level"] = level
    if pbs:
        pbs_jobid = os.environ.get("PBS_JOBID")
        if pbs_jobid:
            pbs_jobid = pbs_jobid.split(".")[0]
            hostname = socket.gethostname()
            for handler, handler_config in config_dict["handlers"].items():
                filename = handler_config.get("filename")
                if filename is not None:
                    parts = list(PurePath(filename).parts)
                    parts[-1] = "_".join(map(str, (pbs_jobid, hostname, parts[-1])))
                    parts.insert(-1, "pbs")
                    new_filename = PurePath(*parts)
                    handler_config["filename"] = str(new_filename)
                    os.makedirs(new_filename.parent, exist_ok=True)
        else:
            logging.config.dictConfig(config_dict)
            logging.getLogger(__name__).warning(
                "pbs=True given, but not running in a PBS job (pbs_jobid empty)."
            )
            return
    logging.config.dictConfig(config_dict)


if __name__ == "__main__":
    import logging

    import cdsapi
    from logging_tree import printout

    c = cdsapi.Client()
    logging.config.dictConfig(LOGGING)
    logger0 = logging.getLogger("")
    logger1 = logging.getLogger(__name__)
    logger2 = logging.getLogger("testing")
    printout()

    for level, level_str in (
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
    ):
        for logger, logger_name in (
            (logger0, "root logger"),
            (logger1, "{} logger".format(__name__)),
            (logger2, "testing logger"),
        ):
            logger.log(level, "{} {}".format(level_str, logger_name + " test"))
