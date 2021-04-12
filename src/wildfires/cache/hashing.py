# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from inspect import iscode

import iris
import joblib
import numpy as np
import pandas as pd
import xxhash


class CodeObj:
    """Return a (somewhat) flattened, hashable version of func.__code__.

    For a function `func`, use like so:
        code_obj = CodeObj(func.__code__).hashable()

    Note that closure variables are not supported.

    """

    expansion_limit = 1000

    def __init__(self, code, __expansion_count=0):
        assert iscode(code), "Must pass in a code object (function.__code__)."
        self.code = code
        self.__expansion_count = __expansion_count

    def hashable(self):
        if self.__expansion_count > self.expansion_limit:
            raise RuntimeError(
                "Maximum number of code object expansions exceeded ({} > {}).".format(
                    self.__expansion_count, self.expansion_limit
                )
            )

        # Get co_ attributes that describe the code object. Ignore the line number and
        # file name of the function definition here, since we don't want unrelated
        # changes to cause a recalculation of a cached result. Changes in comments are
        # ignored, but changes in the docstring will still causes comparisons to fail
        # (this could be ignored as well, however)!
        self.code_dict = OrderedDict(
            (attr, getattr(self.code, attr))
            for attr in dir(self.code)
            if "co_" in attr
            and "co_firstlineno" not in attr
            and "co_filename" not in attr
        )
        # Replace any nested code object (eg. for list comprehensions) with a reduced
        # version by calling the hashable function recursively.
        new_co_consts = []
        for value in self.code_dict["co_consts"]:
            if iscode(value):
                self.__expansion_count += 1
                value = type(self)(value, self.__expansion_count).hashable()
            new_co_consts.append(value)

        self.code_dict["co_consts"] = tuple(new_co_consts)

        return tuple(self.code_dict.values())


class Hasher(ABC):
    @staticmethod
    @abstractmethod
    def test_argument(arg):
        """Determine whether this Hasher is applicable for the given object."""

    @staticmethod
    @abstractmethod
    def hash(x):
        """Calculate the hash value of the given object."""


class MAHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        return isinstance(arg, np.ma.core.MaskedArray)

    @staticmethod
    def hash(x):
        """Compute the hash for a numpy MaskedArray."""
        return xxhash.xxh64_hexdigest(x.data) + xxhash.xxh64_hexdigest(x.mask)


_ma_hasher = MAHasher()


class CubeHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        return isinstance(arg, iris.cube.Cube)

    @staticmethod
    def hash(cube):
        cube_hash = ""
        if isinstance(cube.data, np.ma.core.MaskedArray):
            cube_hash += _ma_hasher.hash(cube.data)
        else:
            cube_hash += xxhash.xxh64_hexdigest(cube.data)
        for coord in cube.coords():
            cube_hash += joblib.hashing.hash(coord)

        cube_hash += joblib.hashing.hash(cube.metadata)
        return cube_hash


_cube_hasher = CubeHasher()


class DatasetHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        from ..data import Dataset

        return isinstance(arg, Dataset)

    @staticmethod
    def hash(dataset):
        """Compute the hash of a Dataset.

        Note: This realises any lazy data.

        """
        # Compute the hash for each piece of data.
        dataset_hash = ""
        for cube in dataset:
            dataset_hash += _cube_hasher.hash(cube)

        return dataset_hash


_dataset_hasher = DatasetHasher()


class DatasetsHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        from ..data import Datasets

        return isinstance(arg, Datasets)

    @staticmethod
    def hash(datasets):
        arg_hash = ""
        for dataset in datasets:
            arg_hash += _dataset_hasher.hash(dataset)
        return arg_hash


class DFHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        return isinstance(arg, pd.DataFrame)

    @staticmethod
    def hash(df):
        """Compute the hash of a pandas DataFrame.

        This only considers the index, data, and column names.

        """
        dataset_hash = xxhash.xxh64_hexdigest(np.ascontiguousarray(df.values))
        dataset_hash += joblib.hashing.hash(df.index)
        dataset_hash += joblib.hashing.hash(df.columns)
        return dataset_hash


@contextmanager
def adjust_n_jobs(arg):
    if hasattr(arg, "n_jobs"):
        # Temporarily set `n_jobs=None` in order to obtain uniform hash values
        # throughout.
        orig_n_jobs = arg.n_jobs
        arg.n_jobs = None
        yield
        # Restore the original value.
        arg.n_jobs = orig_n_jobs
    else:
        # Do nothing.
        yield


# Run outside of context managers.
_default_initial_hashers = []
# Context managers that temporarily change objects to enable consistent hashing.
_default_context_managers = [adjust_n_jobs]
# Run within context managers.
_default_guarded_hashers = [
    MAHasher(),
    DatasetsHasher(),
    DatasetHasher(),
    CubeHasher(),
    DFHasher(),
]
