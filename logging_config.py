#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s:%(levelname)-8s:%(name)-15s:%(message)s',
            },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': os.path.join(log_dir, 'log.log',),
            'mode': 'a',
        }
    },
    'loggers': {
        'test': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }
}
LOGGING['loggers']['__main__'] = LOGGING['loggers']['test']

