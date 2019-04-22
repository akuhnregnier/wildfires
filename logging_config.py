#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


log_dir = os.path.expanduser(os.path.join("~", "Documents", "wildfire_logs"))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": (
                "%(asctime)s:%(levelname)-8s:%(name)-20s"
                ":%(lineno)-5s:%(funcName)-30s:%(message)s"
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
            "interval": "7",
            "level": "DEBUG",
            "formatter": "default",
            "filename": os.path.join(log_dir, "wildfires_debug.log"),
            "mode": "a",
        },
        "info_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": "7",
            "level": "INFO",
            "formatter": "default",
            "filename": os.path.join(log_dir, "wildfires_info.log"),
            "mode": "a",
        },
        "warning_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": "7",
            "level": "WARNING",
            "formatter": "default",
            "filename": os.path.join(log_dir, "wildfires_warning.log"),
            "mode": "a",
        },
        "root_debug_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": "7",
            "level": "DEBUG",
            "formatter": "default",
            "filename": os.path.join(log_dir, "root_debug.log"),
            "mode": "a",
        },
        "root_info_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": "7",
            "level": "INFO",
            "formatter": "default",
            "filename": os.path.join(log_dir, "root_info.log"),
            "mode": "a",
        },
        "root_warning_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": "7",
            "level": "WARNING",
            "formatter": "default",
            "filename": os.path.join(log_dir, "root_warning.log"),
            "mode": "a",
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


if __name__ == "__main__":
    import logging
    import logging.config
    from logging_tree import printout
    import cdsapi

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
