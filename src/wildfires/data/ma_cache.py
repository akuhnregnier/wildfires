# -*- coding: utf-8 -*-
"""Joblib Memory decorator with limited MaskedArray support."""
from functools import wraps

import joblib
import numpy as np
import xxhash

from ..joblib.caching import CodeObj, wrap_decorator
from .cube_aggregation import Datasets
from .datasets import Dataset

__all__ = ("get_hash", "ma_cache")


def hash_ma(x):
    """Compute the hash for a numpy MaskedArray."""
    return xxhash.xxh64_hexdigest(x.data) + xxhash.xxh64_hexdigest(x.mask)


def hash_dataset(dataset):
    """Compute the hash of a Dataset.

    Note: This realises any lazy data.

    """
    # Compute the hash for each piece of data.
    dataset_hash = ""
    for cube in dataset:
        if isinstance(cube.data, np.ma.core.MaskedArray):
            dataset_hash += hash_ma(cube.data)
        else:
            dataset_hash += xxhash.xxh64_hexdigest(cube.data)

    # Finally consider the coordinates and metadata.
    dataset_hash += joblib.hashing.hash(dataset._shallow)

    return dataset_hash


def get_hash(arg):
    """Compute the hash."""
    if isinstance(arg, np.ma.core.MaskedArray):
        arg_hash = hash_ma(arg)
    elif isinstance(arg, Datasets):
        arg_hash = ""
        for dataset in arg:
            arg_hash += hash_dataset(dataset)
    elif isinstance(arg, Dataset):
        arg_hash = hash_dataset(arg)
    else:
        arg_hash = joblib.hashing.hash(arg)
    return arg_hash


@wrap_decorator
def ma_cache(func, *, memory=None):
    """MaskedArray-capable Joblib Memory decorator.

    This is achieved by looking for MaskedArray instances in certain predefined
    locations (e.g. within any Dataset object in `args`) and calculating the hash of
    the data and mask separately before telling Joblib to ignore this object.

    Note: This realises any lazy data.

    """
    if memory is None:
        raise ValueError("A Joblib.memory.Memory instance must be given.")

    @wraps(func)
    def cached_func(*orig_args, **orig_kwargs):
        # Go through the original arguments and hash the contents manually.
        args_hashes = []
        for arg in orig_args:
            args_hashes.append(get_hash(arg))

        # Repeat the above process for the kwargs. The keys should never include
        # MaskedArray data so we only need to deal with the values.
        kwargs_hashes = {}
        for key, arg in orig_kwargs.items():
            kwargs_hashes[key] = get_hash(arg)

        # Include a hashed representation of the original function to ensure we can
        # tell different functions apart.
        func_code = CodeObj(func.__code__).hashable()

        @memory.cache(ignore=["args", "kwargs"])
        def inner(func_code, args_hashes, kwargs_hashes, args, kwargs):
            return func(*args, **kwargs)

        return inner(func_code, args_hashes, kwargs_hashes, orig_args, orig_kwargs)

    return cached_func
