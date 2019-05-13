#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from joblib import Memory

from wildfires.data.datasets import DATA_DIR, HYDE

mem = Memory(location=DATA_DIR, verbose=2)


@mem.cache
def cached_func(dataset, a):
    print("input:", dataset)
    if a == 10:
        return -1
    return 10 * 10


if __name__ == "__main__":
    print(cached_func(HYDE, 3))
