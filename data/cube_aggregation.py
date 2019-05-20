#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate aggregated cubes.

A subset of variables is selected. This selected subset, its total time average, and
its monthly climatology are all stored as pickle files.

TODO: Enable regex based processing of dataset names too (e.g. for
TODO: Selection.remove_datasets or Selection.select_datasets).

"""
import logging
import logging.config
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import wraps
from inspect import iscode
from pprint import pformat, pprint

import iris
import iris.coord_categorisation
from joblib import Memory
from tqdm import tqdm

import wildfires.data.datasets as wildfire_datasets
from wildfires.data.datasets import DATA_DIR, dataset_times
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)
# FIXME: Reduce verbosity!
memory = Memory(location=DATA_DIR, verbose=0)

# FIXME: Use Dataset.pretty and Dataset.pretty_variable_names attributes!!!


def contains(entry, items, exact=True, str_only=False, single_item_type=str):
    """String matching for selected fields.

    Args:
        entry (namedtuple): Tuple wherein existence of `item` is checked.
        item: Item to look for.
        exact (bool): If True, only accept exact matches for strings,
            ie. replace `item` with `^item$` before using `re.search`.
        str_only (bool): If True, only compare strings and ignore other items in
            `entry`.

    """
    if isinstance(items, single_item_type):
        items = (items,)
    for item in items:
        for i, value in enumerate(entry):
            if hasattr(entry, "_fields"):
                value = getattr(entry, entry._fields[i])
            # If they are both strings, use regular expressions.
            if all(isinstance(obj, str) for obj in (item, value)):
                if exact:
                    item = "^" + item + "$"
                if re.search(item, value):
                    return True
            elif not str_only:
                if item == value:
                    return True
    return False


class Datasets:
    """Keep track of datasets and associated variables.

    Both original (raw) and user-friendly (pretty) versions of variable names are
    recorded. Every dataset's raw and pretty variable names are guaranteed to be
    unique amongst all raw and pretty variable names in that dataset (this is not
    guaranteed amongst all datasets!).

    A raw and pretty name is also stored for each dataset. These names are guaranteed
    to be unique amongst all raw and pretty dataset names in the selection.

    Examples:
        >>> from .datasets import HYDE
        >>> from ..tests.test_datasets import data_is_available
        >>> instance_sel = Datasets()
        >>> if data_is_available:
        ...     sel = Datasets().add(HYDE())
        ...     assert "popd" in sel.raw_variable_names

    """

    def __init__(self, datasets=None):
        self.datasets = []
        if datasets is not None:
            if isinstance(datasets, wildfire_datasets.Dataset):
                self.add(datasets)
            else:
                for dataset in datasets:
                    self.add(dataset)

    def __len__(self):
        return len(self.datasets)

    def __eq__(self, other):
        if isinstance(other, Datasets):
            return sorted(self.datasets, key=lambda x: x.name) == sorted(
                other.datasets, key=lambda x: x.name
            )
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Datasets):
            other = other.datasets
        elif isinstance(other, wildfire_datasets.Dataset):
            other = (other,)

        new_datasets = deepcopy(self)
        for dataset in other:
            new_datasets.add(dataset)
        return new_datasets

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Datasets):
            other = other.datasets
        elif isinstance(other, wildfire_datasets.Dataset):
            other = (other,)

        for dataset in other:
            self.add(dataset)
        return self

    def __str__(self):
        return pformat(self.get("all", "pretty"))

    def __repr__(self):
        return pformat(self.datasets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self.datasets[index])
        if isinstance(index, str):
            new_index = self.get_index(index)
        elif isinstance(index, wildfire_datasets.Dataset):
            new_index = self.datasets.index(index)
        else:
            new_index = index
        return self.datasets[new_index]

    def copy(self):
        return deepcopy(self)

    @property
    def raw_variable_names(self):
        """Return a tuple of all raw variable names."""
        return tuple(
            name for names in self.get("all", "raw").values() for name in names
        )

    @property
    def pretty_variable_names(self):
        """Return a tuple of all pretty variable names."""
        return tuple(
            name for names in self.get("all", "pretty").values() for name in names
        )

    @property
    def raw_dataset_names(self):
        """Return a tuple of all raw dataset names."""
        return tuple(self.get("all", "raw"))

    @property
    def pretty_dataset_names(self):
        """Return a tuple of all pretty dataset names."""
        return tuple(self.get("all", "pretty"))

    @property
    def cubes(self):
        """Return all cubes."""
        return iris.cube.CubeList(
            cube for dataset in self.datasets for cube in dataset.cubes
        )

    def get_index(self, dataset_name):
        """Get dataset index from a dataset name.

        The given name may match either the raw or the pretty name.

        """
        if not isinstance(dataset_name, str):
            dataset_name = dataset_name[0]
        for dataset in self.datasets:
            if dataset_name in dataset.names():
                return self.datasets.index(dataset)
        raise ValueError("Dataset name '{}' not found.".format(dataset_name))

    def add(self, dataset):
        """Add a dataset to the database.

        Args:
            dataset (`Dataset`):

        Raises:
            ValueError: If the provided dataset matches an existing dataset.

        """
        for stored_dataset in self.datasets:
            if set(stored_dataset.names()).intersection(set(dataset.names())):
                raise ValueError(
                    "Match for datasets '{}' and '{}'.".format(stored_dataset, dataset)
                )

        self.datasets.append(dataset)
        return self

    def get(self, dataset_name="all", variable_format="raw"):
        """Return a dictionary representation of datasets and variables.

        Args:
            dataset (str): Name of the dataset, which may match either the raw or
                pretty dataset name. If "all", get information for all datasets.
            variable_format (str): If "raw", get tuples of the names of the variable
                as returned by `cube.name()`. If "pretty", get tuples of formatted
                variable names, eg. for display on figures. If "all", get tuples
                containing both (ie. tuples of tuples of str).

        Raises:
            AttributeError: If `variable_format` does not match one of the known
                formats "raw", "pretty", or "all".

        Returns:
            dict: Mapping from the chosen dataset(s) to the variable(s) contained
                therein.

        """
        logger.debug("Get() called with Datasets '{}'".format(id(self)))

        if dataset_name == "all":
            index = slice(None)
        else:
            index = self.get_index(dataset_name)
            index = slice(index, index + 1)

        to_format = self[index]
        formatted = OrderedDict()

        for dataset in to_format:
            formatted[dataset.names(variable_format)] = dataset.variable_names(
                variable_format
            )

        return formatted

    def process_datasets(self, selection, names, operation="select"):
        """Process datasets in `selection` which match `names`.

        Args:
            selection (`Datasets`): Selection containing datasets to process.
            names (str or iterable): A name or series of names (dataset_name, ...),
                for which a match with either raw or pretty names is sufficient.
            operation (str): If "select", add matching datasets to `selection`. If
                "remove", remove matching datasets from `selection`.

        Raises:
            KeyError: If a dataset is not found.

        Returns:
            `Datasets`: `selection` processed using `processing_func`.
        """
        if isinstance(names, str):
            names = (names,)
        new_datasets = []
        for search_dataset in names:
            for stored_dataset in self.datasets:
                if search_dataset in stored_dataset.names():
                    if operation == "select":
                        new_datasets.append(stored_dataset)
                    elif operation == "remove":
                        selection.datasets.remove(stored_dataset)
                    else:
                        raise ValueError("Invalid operation '{}'".format(operation))
                    break
            else:
                raise KeyError("Dataset '{}' not found.".format(search_dataset))

        if operation == "select":
            # Do this here since `selection` could be `self` as well.
            selection.datasets[:] = new_datasets
        return selection

    def select_datasets(self, names, inplace=True):
        """Return a new `Datasets` containing only datasets matching `names`.

        Args:
            names (str or iterable): A name or series of names (dataset_name, ...),
                for which a match with either raw or pretty names is sufficient.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection before removing entries.

        Raises:
            KeyError: If a dataset is not found.

        Returns:
            `Datasets`: A copy of a subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = type(self)()
        return self.process_datasets(selection, names, operation="select")

    def remove_datasets(self, names, inplace=True):
        """Remove datasets not matching `names`.

        Args:
            names (str or iterable): A name or series of names (dataset_name, ...),
                for which a match with either raw or pretty names is sufficient.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection before removing entries.

        Raises:
            KeyError: If a dataset is not found.

        Returns:
            `Datasets`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = deepcopy(self)
        return self.process_datasets(selection, names, operation="remove")

    @staticmethod
    def __remove_variable(selection, dataset, cube_name):
        """Remove a dataset's cube in-place."""
        if not isinstance(dataset, str):
            dataset = dataset[0]
        if not isinstance(cube_name, str):
            cube_name = cube_name[0]
        assert isinstance(dataset, str)
        assert isinstance(cube_name, str)

        def selection_func(cube):
            return cube.name() != cube_name

        selection[dataset].cubes = selection[dataset].cubes.extract(
            iris.Constraint(cube_func=selection_func)
        )

    def dict_process_variables(
        self, selection, search_dict, which="remove", exact=True
    ):
        """Modify the selection's matching variable entries using the given function.

        Args:
            selection (`Datasets`): Selection containing variables to process.
            search_dict (dict): A dict with form
                  {dataset_name: (variable_name, ...), ...}, where `dataset_name` and
                  `variable_name` are of type `str`. This is used to select the
                  variables to process.
            which (str): If "remove", remove variables in-place. If "add", add
                variables.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Returns:
            `selection` processed using `processing_func`.

        """
        assert which in {"remove", "add"}
        removal_dict = defaultdict(list)
        # Look for a match for each desired variable.
        for search_dataset, search_variable in (
            (search_dataset, search_variable)
            for search_dataset, target_variables in search_dict.items()
            for search_variable in target_variables
        ):
            matches = 0
            for stored_dataset, stored_variable in (
                (stored_datasets, stored_variable)
                for stored_datasets, stored_variables in self.get(
                    search_dataset, variable_format="all"
                ).items()
                for stored_variable in stored_variables
            ):
                if which == "remove" and contains(
                    stored_variable, search_variable, exact=exact
                ):
                    self.__remove_variable(selection, stored_dataset, stored_variable)
                    # Move on to the next desired variable.
                    break
                if which == "add":
                    if contains(stored_variable, search_variable, exact=exact):
                        matches += 1
                    else:
                        removal_dict[stored_dataset].append(stored_variable)
            else:
                if which == "remove" or which == "add" and matches < 1:
                    raise KeyError(
                        "Variable '{}' not found for dataset '{}'.".format(
                            search_variable, search_dataset
                        )
                    )
        if which == "add":
            return self.dict_process_variables(
                selection, removal_dict, "remove", exact=exact
            )
        elif which == "remove":
            # Remove empty datasets.
            selection.datasets[:] = [
                dataset for dataset in selection.datasets if dataset
            ]
        return selection

    def dict_select_variables(self, search_dict, inplace=True, exact=True):
        """Return a new `Selection` containing only variables matching `search_dict`.

        Args:
            search_dict (dict): A dict with form
                  {dataset_name: (variable_name, ...), ...}, where `dataset_name` and
                  `variable_name` are of type `str`. This is used to select the
                  variables to process.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but only containing the selected
                entries. If False, make a copy of the selection before removing
                entries.

        Raises:
            KeyError: If a key or value in `search_dict` is not found in the
                database.

        Returns:
            `Selection`: A copy of a subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = type(self)()
        return self.dict_process_variables(selection, search_dict, "add", exact=exact)

    def dict_remove_variables(self, search_dict, inplace=True, exact=True):
        """Remove variables matching `search_dict`.

        Cube pruning is performed after removal of entries to remove redundant cubes
        from dataset instances.

        Args:
            search_dict (dict): A dict with form
                  {dataset_name: (variable_name, ...), ...}, where `dataset_name` and
                  `variable_name` are of type `str`. This is used to select the
                  variables to process.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection before removing entries.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Raises:
            KeyError: If a key or value in `search_dict` is not found in the
                database.

        Returns:
            `Datasets`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = deepcopy(self)
        selection = self.dict_process_variables(
            selection, search_dict, "remove", exact=exact
        )
        return selection

    def process_variables(self, selection, names, which="remove", exact=True):
        """Modify the selection's matching variable entries using the given function.

        Args:
            selection (`Datasets`): Selection containing variables to process.
            names (str or iterable): A name or series of names (variable_name, ...),
                for which a match with either raw or pretty names is sufficient. This
                is used to determine which variables to process.
            which (str): If "remove", remove variables in-place. If "add", add
                variables.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Raises:
            ValueError: If raw and pretty names are not unique across all stored
                variable names (raw and pretty) for all datasets. In this case, select
                variables using a dictionary instead of an iterable.
            KeyError: If a variable is not found in the database.

        Returns:
            `Datasets`: `selection` processed using `processing_func`.

        """
        assert which in {"remove", "add"}
        removal_dict = defaultdict(list)
        if isinstance(names, str):
            names = (names,)
        unpacked_datasets_variables = tuple(
            (dataset[0], variable_name)
            for dataset, variables in self.get("all", "all").items()
            for variable_names in variables
            # Use set() here, since the pretty variable name could be the same as the
            # raw variable name if no pretty variable has been provided. This would
            # result in a false positive, as we are only interested in identical names
            # for different variables.
            for variable_name in set(variable_names)
        )
        all_variable_names = [
            variable_name
            for (dataset_name, variable_name) in unpacked_datasets_variables
        ]
        n_duplicates = len(all_variable_names) - len(set(all_variable_names))
        if n_duplicates:
            duplicated_variables = []
            duplicated_datasets = []
            for variable_name in all_variable_names:
                if (
                    variable_name not in duplicated_variables
                    and all_variable_names.count(variable_name) > 1
                ):
                    duplicated_variables.append(variable_name)
                    for dataset_name, comp_variable_name in unpacked_datasets_variables:
                        if variable_name == comp_variable_name:
                            duplicated_datasets.append(dataset_name)
            duplicated_variables = set(duplicated_variables)
            duplicated_datasets = set(duplicated_datasets)
            logger.warning("Duplicated variable names: {}".format(duplicated_variables))
            logger.warning("Duplicated datasets: {}".format(duplicated_datasets))
            raise ValueError(
                "Raw and pretty variable names are not unique ({} duplicate(s)). Use "
                "dictionaries for selection. or remove some of the datasets {}.".format(
                    n_duplicates, duplicated_datasets
                )
            )

        # Look for a match for each desired variable.
        logger.debug("Selecting the following: '{}'.".format(names))
        for search_variable in names:
            matches = 0
            for stored_dataset, stored_variable in (
                (stored_dataset_names, stored_variable_names)
                for stored_dataset_names, stored_variables in (
                    self.get("all", "all").items()
                )
                for stored_variable_names in stored_variables
            ):
                # Check against all variables in the dataset.
                if which == "remove" and contains(
                    stored_variable, search_variable, exact=exact
                ):
                    logger.debug(
                        "Selecting '{}: {}'.".format(stored_dataset, stored_variable)
                    )
                    self.__remove_variable(selection, stored_dataset, stored_variable)
                    # Move on to the next variable.
                    break
                if which == "add":
                    if contains(stored_variable, search_variable, exact=exact):
                        matches += 1
                    else:
                        removal_dict[stored_dataset].append(stored_variable)
            else:
                if which == "remove" or which == "add" and matches < 1:
                    raise KeyError("Variable '{}' not found.".format(search_variable))

        if which == "add":
            return self.dict_process_variables(
                selection, removal_dict, "remove", exact=exact
            )
        elif which == "remove":
            # Remove empty datasets.
            selection.datasets[:] = [
                dataset for dataset in selection.datasets if dataset
            ]
        return selection

    def select_variables(self, names, inplace=True, exact=True):
        """Return a new `Datasets` containing only variables matching `criteria`.

        Args:
            names (str or iterable): A name or series of names (variable_name, ...),
                for which a match with either raw or pretty names is sufficient. This
                is used to determine which variables to process.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but only containing the selected
                entries. If False, make a copy of the selection before removing
                entries.

        Raises:
            ValueError: If raw and pretty names are not unique across all stored
                variable names (raw and pretty) for all datasets. In this case, select
                variables using a dictionary instead of an iterable.
            KeyError: If a variable is not found in the database.

        Returns:
            `Datasets`: A copy of a subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = deepcopy(self)
        return self.process_variables(selection, names, "add", exact=exact)

    def remove_variables(self, names, inplace=True, exact=True):
        """Return a new `Datasets` without the variables matching `criteria`.

        Cube pruning is performed after removal of entries to remove redundant cubes
        from dataset instances.

        Args:
            names (str or iterable): A name or series of names (variable_name, ...),
                for which a match with either raw or pretty names is sufficient. This
                is used to determine which variables to process.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                object as the original selection, but without the removed entries. If
                False, make a copy of the selection before removing entries.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Raises:
            ValueError: If raw and pretty names are not unique across all stored
                variable names (raw and pretty) for all datasets. In this case, select
                variables using a dictionary instead of an iterable.
            KeyError: If a variable is not found in the database.

        Returns:
            `Datasets`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = deepcopy(self)
        selection = self.process_variables(selection, names, "remove", exact=exact)
        return selection

    def show(self, variable_format="all"):
        """Print out a representation of the selection."""
        pprint(self.get(dataset_name="all", variable_format=variable_format))


def get_all_datasets(pretty_dataset_names=None, pretty_variable_names=None):
    """Get all valid datasets defined in the `wildfires.data.datasets` module.

    Args:
        pretty_variable_names (dict): Dictionary mapping raw to pretty variable names
            ({raw: pretty, ...}).

    Returns:
        `Datasets`: Selection object describing the datasets.

    """
    # TODO: Implement pretty dataset and variable names.

    if pretty_dataset_names is None:
        pretty_dataset_names = {}
    if pretty_variable_names is None:
        pretty_variable_names = {}
    selection = Datasets()
    dataset_names = dir(wildfire_datasets)
    for dataset_name in dataset_names:
        logger.debug("Testing if {} is a valid Dataset.".format(dataset_name))
        obj = getattr(wildfire_datasets, dataset_name)
        if (
            obj != wildfire_datasets.Dataset
            and hasattr(obj, "__mro__")
            and wildfire_datasets.Dataset in obj.__mro__
        ):
            try:
                dataset = obj()
                selection.add(dataset)
            except NotImplementedError:
                logger.info("{} is not implemented.".format(dataset_name))

    return selection


def wrap_decorator(decorator):
    @wraps(decorator)
    def new_decorator(*args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return decorator(args[0])

        def decorator_wrapper(f):
            return decorator(f, *args, **kwargs)

        return decorator_wrapper

    return new_decorator


class CodeObj:
    """Return a (somewhat) flattened, hashable version of func.__code__.

    For a function `func`, use like so:
        code_obj = CodeObj(func.__code__).hashable()

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

        # Get co_ attributes that describe the code object.
        self.code_dict = OrderedDict(
            (attr, getattr(self.code, attr)) for attr in dir(self.code) if "co_" in attr
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


@wrap_decorator
def clever_cache(func):
    # NOTE: There is a known bug preventing joblib from pickling numpy MaskedArray!
    # NOTE: https://github.com/joblib/joblib/issues/573
    # NOTE: We will avoid this bug by replacing Dataset instances (which may hold
    # NOTE: references to masked arrays) with their (shallow) immutable string
    # NOTE: representations.
    """Circumvent bug preventing joblib from pickling numpy MaskedArray instances.

    This applies to MaskedArray in the input arguments only.

    Do this by giving joblib a different version of the input arguments to cache,
    while still passing the normal arguments to the decorated function.

    """

    @wraps(func)
    def takes_original_selection(*args, **kwargs):
        """Function that is visible to the outside."""
        if not isinstance(args[0], Datasets) or any(
            isinstance(arg, Datasets)
            for arg in list(args[1:]) + list(kwargs.values()) + list(kwargs.keys())
        ):
            raise TypeError(
                "The first positional argument, and only the first argument "
                "should be a `Datasets` instance."
            )
        original_selection = args[0]
        string_representation = str(original_selection.get("all", "raw"))

        func_code = CodeObj(func.__code__).hashable()

        @memory.cache(ignore=["original_selection"])
        def takes_split_selection(
            func_code, string_representation, original_selection, *args, **kwargs
        ):
            out = func(original_selection, *args, **kwargs)
            return out

        return takes_split_selection(func_code, string_representation, *args, **kwargs)

    return takes_original_selection


def print_datasets_dates(selection):
    min_time, max_time, times_df = dataset_times(selection.datasets)
    print(times_df)


@clever_cache
def prepare_selection(selection):
    """Prepare cubes matching the given selection for analysis.

    Calculate 3 different averages at the same time to avoid repeat loading.

    Args:
        selection (`Datasets`): Selection specifying the variables to use.
        averaging (str or None): If None, do not perform any averaging. If "mean",
            average original monthly data over all time. If "monthly climatology",
            produce a monthly climatology.

    """
    min_time, max_time, times_df = dataset_times(selection.datasets)

    # Limit the amount of data that has to be processed.
    logger.info("Limiting data")
    for dataset in selection:
        dataset.limit_months(min_time, max_time)
    logger.info("Finished limiting data")

    # Regrid cubes to the same lat-lon grid.
    # TODO: change lat and lon limits and also the number of points!!
    # Always work in 0.25 degree steps? From the same starting point?
    logger.info("Starting regridding of all datasets")
    for dataset in selection:
        dataset.regrid()
    logger.info("Finished regridding of all datasets.")

    logger.info("Starting temporal upscaling.")
    raw_datasets = Datasets()
    for dataset in tqdm(selection):
        # TODO: Inplace argument for get_mothly_data methods?
        monthly_dataset = deepcopy(dataset)
        monthly_dataset.cubes[:] = monthly_dataset.get_monthly_data(min_time, max_time)
        raw_datasets += monthly_dataset

    # logger.info("Finished temporal upscaling.")
    # iris.save(cubes, TARGET_FILES[0])

    logger.info("Calculating total mean.")
    mean_datasets = Datasets()
    for dataset in tqdm(selection):
        mean_dataset = deepcopy(dataset)
        for i, cube in enumerate(mean_dataset):
            # TODO: Implement __setitem__ for Dataset to make this cleaner.
            mean_dataset.cubes[i] = cube.collapsed("time", iris.analysis.MEAN)
        mean_datasets += mean_dataset

    # iris.save(cubes, TARGET_FILES[1])
    # logger.info(mean_cubes)

    logger.info("Calculating monthly climatology.")
    climatology_datasets = Datasets()
    for dataset in tqdm(selection):
        climatology = deepcopy(dataset)
        for cube in dataset:
            if not cube.coords("month_number"):
                iris.coord_categorisation.add_month_number(cube, "time")
            climatology.cubes[i] = cube.aggregated_by(
                "month_number", iris.analysis.MEAN
            )
        climatology_datasets += climatology

    # iris.save(cubes, TARGET_FILES[2])

    return raw_datasets, mean_datasets, climatology_datasets


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)

    selection = get_all_datasets()

    selection.remove_datasets(
        (
            "AvitabileAGB",
            "ESA_CCI_Landcover",
            "GFEDv4s",
            "GPW_v4_pop_dens",
            "Thurner_AGB",
        )
    )
    selected_names = [
        "AGBtree",
        # 'mean temperature',
        # 'monthly precipitation',
        "maximum temperature",
        "minimum temperature",
        # 'Quality Flag with T=1',
        # 'Soil Water Index with T=60',
        # 'Soil Water Index with T=5',
        # 'Quality Flag with T=60',
        # 'Soil Water Index with T=20',
        # 'Soil Water Index with T=40',
        # 'Quality Flag with T=20',
        # 'Surface State Flag',
        "Soil Water Index with T=1",
        # 'Quality Flag with T=100',
        # 'Soil Water Index with T=15',
        # 'Quality Flag with T=10',
        # 'Soil Water Index with T=10',
        # 'Quality Flag with T=40',
        # 'Quality Flag with T=15',
        # 'Quality Flag with T=5',
        "Soil Water Index with T=100",
        "ShrubAll",
        "TreeAll",
        "pftBare",
        "pftCrop",
        "pftHerb",
        "pftNoLand",
        # 'pftShrubBD',
        # 'pftShrubBE',
        # 'pftShrubNE',
        # 'pftTreeBD',
        # 'pftTreeBE',
        # 'pftTreeND',
        # 'pftTreeNE',
        "monthly burned area",
        "dry_days",
        "dry_day_period",
        "precip",
        "SIF",
        # 'cropland',
        # 'ir_norice',
        # 'rf_norice',
        "popd",
        # 'conv_rangeland',
        # 'rangeland',
        # 'tot_rainfed',
        # 'pasture',
        # 'rurc',
        # 'rf_rice',
        # 'tot_rice',
        # 'uopp',
        # 'popc',
        # 'ir_rice',
        # 'urbc',
        # 'grazing',
        # 'tot_irri',
        "Combined Flash Rate Time Series",
        "VODorig",
        # 'Standard Deviation of LAI',
        "Fraction of Absorbed Photosynthetically Active Radiation",
        "Leaf Area Index",
        # 'Standard Deviation of fPAR',
        # 'Simard_Pinto_3DGlobalVeg_L3C',
        # 'biomass_totalag',
        # 'biomass_branches',
        # 'biomass_foliage',
        # 'biomass_roots',
        # 'biomass_stem'
    ]
    # FIXME: Make this operation inplace by default?
    selection = selection.select_variables(selected_names)
    selection.prune_cubes()

    assert len(selection.cubes) == len(
        selected_names
    ), "There should be as many cube as selected variables."

    selection.show("pretty")
    print_datasets_dates(selection)
    aggregated_cubes = prepare_selection(selection)
