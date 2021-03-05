# -*- coding: utf-8 -*-
"""Joblib Memory decorator with limited MaskedArray support."""
from functools import partial, wraps

import joblib
import numpy as np
import pandas as pd
import xxhash

from ..joblib.caching import CodeObj
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


def hash_df(df):
    """Compute the hash of a pandas DataFrame.

    This only considers the index, data, and column names.

    """
    dataset_hash = xxhash.xxh64_hexdigest(df.values)
    dataset_hash += joblib.hashing.hash(df.index)
    dataset_hash += joblib.hashing.hash(df.columns)
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
    elif isinstance(arg, pd.DataFrame):
        arg_hash = hash_df(arg)
    else:
        arg_hash = joblib.hashing.hash(arg)
    return arg_hash


class ma_cache:
    """MaskedArray-capable Joblib Memory decorator.

    This is achieved by looking for MaskedArray instances in certain predefined
    locations (e.g. within any Dataset object in `args`) and calculating the hash of
    the data and mask separately before telling Joblib to ignore this object.

    Note: This realises any lazy data.

    """

    def __init__(self, *, memory=None, hash_func=get_hash):
        if memory is None:
            raise ValueError("A Joblib.memory.Memory instance must be given.")
        self.memory = memory
        self.hash_func = hash_func

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            # The decorator was not configured with additional arguments.
            return self._decorator(args[0])

        def decorator_wrapper(f):
            return self._decorator(f, *args, **kwargs)

        return decorator_wrapper

    def _get_hashed(self, func, *args, **kwargs):
        # Go through the original arguments and hash the contents manually.
        args_hashes = []
        for arg in args:
            args_hashes.append(self.hash_func(arg))

        # Repeat the above process for the kwargs. The keys should never include
        # MaskedArray data so we only need to deal with the values.
        kwargs_hashes = {}
        for key, arg in kwargs.items():
            kwargs_hashes[key] = self.hash_func(arg)

        # Include a hashed representation of the original function to ensure we can
        # tell different functions apart.
        func_code = CodeObj(func.__code__).hashable()

        return dict(
            func_code=func_code, args_hashes=args_hashes, kwargs_hashes=kwargs_hashes
        )

    def _decorator(self, func):
        def inner(hashed, args, kwargs):
            return func(*args, **kwargs)

        cached_inner = self.memory.cache(ignore=["args", "kwargs"])(inner)

        @wraps(func)
        def cached_func(*orig_args, **orig_kwargs):
            hashed = self._get_hashed(func, *orig_args, **orig_kwargs)
            return cached_inner(hashed, orig_args, orig_kwargs)

        return Decorated(
            cached_func=cached_func,
            cached_inner=cached_inner,
            bound_get_hashed=partial(self._get_hashed, func),
        )


class Decorated:
    """A cached function."""

    def __init__(self, cached_func, cached_inner, bound_get_hashed):
        self.cached_func = cached_func
        self._cached_inner = cached_inner
        self._get_hashed = bound_get_hashed

    def __call__(self, *args, **kwargs):
        return self.cached_func(*args, **kwargs)

    def is_cached(self, *args, **kwargs):
        """Return True if this call is already cached and False otherwise."""
        return self._cached_inner.store_backend.contains_item(
            self._cached_inner._get_output_identifiers(
                self._get_hashed(*args, **kwargs), args, kwargs
            )
        )
