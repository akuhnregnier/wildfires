#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


DATA_DIR = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.pardir,
        'Data'
        ))

GFED4_DIR = os.path.join(DATA_DIR, 'gfed4')

SOIL_MOIST_DIR = os.path.join(DATA_DIR, 'soil-moisture')


if __name__ == '__main__':
    print('Data dir')
    print(DATA_DIR)
