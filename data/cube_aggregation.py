#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate aggregated cubes.

A subset of variables is selected. This selected subset, its total time average, and
its monthly climatology are all stored as pickle files.

TODO: Enable regex based processing of dataset names too (e.g. for
TODO: Selection.remove_datasets or Selection.select_datasets).

"""
import inspect
import logging
import logging.config
import re
from collections import OrderedDict
from copy import deepcopy
from functools import partial, reduce, wraps
from pprint import pformat, pprint

import iris
import iris.coord_categorisation
import numpy as np

import wildfires.data.datasets as wildfire_datasets
from joblib import Memory, Parallel, delayed
from wildfires.data.datasets import (
    DATA_DIR,
    data_is_available,
    dataset_times,
    fill_dataset,
    get_climatology,
    get_mean,
    get_monthly,
    get_monthly_mean_climatology,
    homogenise_cube_mask,
)
from wildfires.joblib.caching import CodeObj, wrap_decorator
from wildfires.logging_config import LOGGING
from wildfires.utils import get_ncpus, match_shape

logger = logging.getLogger(__name__)
memory = Memory(location=DATA_DIR if data_is_available() else None, verbose=1)

IGNORED_DATASETS = [
    "AvitabileAGB",
    "CRU",
    "ESA_CCI_Fire",
    "ESA_CCI_Landcover",
    "ESA_CCI_Soilmoisture",
    "ESA_CCI_Soilmoisture_Daily",
    "GFEDv4s",
    "GPW_v4_pop_dens",
    "LIS_OTD_lightning_time_series",
    "Simard_canopyheight",
    "Thurner_AGB",
]

# TODO: Use Dataset.pretty and Dataset.pretty_variable_names attributes!!!


@wrap_decorator
def datasets_cache(func):
    # NOTE: There is a known bug preventing joblib from pickling numpy MaskedArray!
    # NOTE: https://github.com/joblib/joblib/issues/573
    # NOTE: We will avoid this bug by replacing Dataset instances (which may hold
    # NOTE: references to masked arrays) with their (shallow) immutable string
    # NOTE: representations.
    """Circumvent bug preventing joblib from pickling numpy MaskedArray instances.

    This applies to MaskedArray in the input arguments only.

    Do this by giving joblib a different version of the input arguments to cache,
    while still passing the normal arguments to the decorated function.

    Note:
        `dataset_function` argument in `func` must be a keyword argument.

    """

    @wraps(func)
    def takes_original_selection(*orig_args, **orig_kwargs):
        """Function that is visible to the outside."""
        if not isinstance(orig_args[0], Datasets) or any(
            isinstance(arg, Datasets)
            for arg in list(orig_args[1:]) + list(orig_kwargs.values())
        ):
            raise TypeError(
                "The first positional argument, and only the first argument "
                f"should be a `Datasets` instance, got '{type(orig_args[0])}' "
                "as the first argument."
            )
        original_selection = orig_args[0]
        string_representation = "\n".join(
            dataset._shallow for dataset in original_selection.datasets
        ) + str(original_selection.get("all", "raw"))

        # Ignore instances with a __call__ method here which also wouldn't necessarily
        # have a __name__ attribute that could be used for sorting!
        functions = [func]
        for param_name, param_value in inspect.signature(func).parameters.items():
            default_value = param_value.default
            # If the default value is a function, and it is not given in `orig_kwargs`
            # already. This is guaranteed to work as long as `func` employs a
            # keyword-only argument for functions like this.
            if (
                param_name not in orig_kwargs
                and default_value is not inspect.Parameter.empty
                and hasattr(default_value, "__code__")
            ):
                functions.append(default_value)
                orig_kwargs[param_name] = default_value

        functions.extend(
            f
            for f in list(orig_args[1:]) + list(orig_kwargs.values())
            if hasattr(f, "__code__")
        )

        functions = list(set(functions))
        functions.sort(key=lambda f: f.__name__)
        func_code = tuple(CodeObj(f.__code__).hashable() for f in functions)

        assert len(func_code) == 2, (
            "Only 2 functions are currently supported. One is the decorated function, "
            "the other is the processing function `dataset_function`."
        )

        @memory.cache(ignore=["original_selection", "dataset_function"])
        def takes_split_selection(
            func_code,
            string_representation,
            original_selection,
            *args,
            dataset_function=None,
            **kwargs,
        ):
            # NOTE: The reason why this works is that the combination of
            # [original_selection] + args here is fed the original `orig_args`
            # iterable. In effect, the `original_selection` argument absorbs one of
            # the elements of `orig_args` in the function call to
            # `takes_split_selection`, so it is not passed in twice. The `*args`
            # parameter above absorbs the rest. This explicit listing of
            # `original_selection` is necessary, as we need to explicitly ignore
            # `original_selection`, which is the whole point of this decorator.
            assert dataset_function is not None
            out = func(
                original_selection, *args, dataset_function=dataset_function, **kwargs
            )
            return out

        return takes_split_selection(
            func_code, string_representation, *orig_args, **orig_kwargs
        )

    return takes_original_selection


def contains(
    stored_items, search_items, exact=True, str_only=False, single_item_type=str
):
    """String matching for selected fields.

    Args:
        stored_items (iterable or namedtuple): Tuple wherein existence of `items` is
            checked.
        search_items: Item(s) to look for.
        exact (bool): If True, only accept exact matches for strings,
            ie. replace `item` with `^item$` before using `re.search`. Additionally,
            `item` is replaced with `re.escape(item)`.
        str_only (bool): If True, only compare strings and ignore other items in
            `entry`.
        single_item_type (type)

    """
    if isinstance(stored_items, single_item_type):
        stored_items = (stored_items,)
    if isinstance(search_items, single_item_type):
        search_items = (search_items,)
    for search_item in search_items:
        for i, stored_item in enumerate(stored_items):
            if hasattr(stored_items, "_fields"):
                stored_item = getattr(stored_items, stored_items._fields[i])
            # If they are both strings, use regular expressions.
            if all(isinstance(obj, str) for obj in (search_item, stored_item)):
                if exact and i == 0:
                    search_item = "^" + re.escape(search_item) + "$"
                if re.search(search_item, stored_item):
                    return True
            elif not str_only:
                if search_item == stored_item:
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
        >>> from .datasets import HYDE, data_is_available
        >>> instance_sel = Datasets()
        >>> if data_is_available():
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

    @property
    def datasets(self):
        # Could perhaps be optimised by replacing `__datasets` with an immutable
        # container, and then sorting it only when it is assigned a new value in the
        # setter method. This could also be applied to the CubeLists in the `Dataset`
        # instances as well.
        self.__datasets = list(
            sorted(self.__datasets, key=lambda dataset: dataset.name)
        )
        return self.__datasets

    @datasets.setter
    def datasets(self, new_datasets):
        self.__datasets = new_datasets

    @property
    def dataset(self):
        """Convenience method to access a single stored dataset."""
        if len(self.datasets) != 1:
            raise ValueError(
                f"Expected 1 Dataset instance, but found {len(self.datasets)} "
                "Dataset instances."
            )
        return self.datasets[0]

    def copy(self, deep=False):
        """Return a copy of the dataset collection.

        Args:
            deep (bool): If True, return a deep copy, which also copies the underlying
                data structures, such as all of the datasets' cubes and their data. If
                False, a shallow copy is made which contains references to the same
                datasets.
        Returns:
            `Datasets`

        """
        if deep:
            return deepcopy(self)
        # Execute the shallow copy just to the level before the individual cubes, ie.
        # make a copy of the `Dataset` instances, as well as their associated cube
        # lists.
        datasets_copy = type(self)()
        datasets_copy.datasets = []
        for original_dataset in self:
            datasets_copy.datasets.append(original_dataset.copy(deep=False))
        return datasets_copy

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

    @property
    def cube(self):
        """Return a single cube, if only one cube is stored in the collection."""
        return self.dataset.cube

    def get_index(self, dataset_name, full_index=False):
        """Get dataset index from a dataset name.

        Args:
            dataset_name (str): This name may match either the raw or the pretty name.
            full_index (bool, optional): If False (default), return the integer index
                of the dataset. Otherwise, return the tuple of dataset names which is
                used to specify datasets in the dict processing methods
                dict_remove_variables() and dict_select_variables().

        """
        if not isinstance(dataset_name, str):
            dataset_name = dataset_name[0]
        for dataset in self.datasets:
            stored_names = dataset.names()
            if contains(stored_names, dataset_name):
                if full_index:
                    return stored_names
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
                    "Matching datasets '{}' and '{}'.".format(stored_dataset, dataset)
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

        logger.debug("Returning formatted dict with {} entries.".format(len(formatted)))

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

    def select_datasets(self, names, inplace=True, copy=False):
        """Return a new `Datasets` containing only datasets matching `names`.

        Args:
            names (str or iterable): A name or series of names (dataset_name, ...),
                for which a match with either raw or pretty names is sufficient.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection (see argument `copy`) before
                removing entries.
            copy (bool, optional): Only applies if `inplace` is False. If False
                (default), the new `Datasets` object will contain references to the
                same `Dataset` instances and associated cubes as the original. If
                True, a deep copy will be made which will also copy the underlying
                data.

        Raises:
            KeyError: If a dataset is not found.

        Returns:
            `Datasets`: A copy of a subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = self.copy()

        output = self.process_datasets(selection, names, operation="select")

        if copy:
            return output.copy(deep=True)
        return output

    def remove_datasets(self, names, inplace=True, copy=False):
        """Remove datasets not matching `names`.

        Args:
            names (str or iterable): A name or series of names (dataset_name, ...),
                for which a match with either raw or pretty names is sufficient.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection (see argument `copy`) before
                removing entries.
            copy (bool, optional): Only applies if `inplace` is False. If False
                (default), the new `Datasets` object will contain references to the
                same `Dataset` instances and associated cubes as the original. If
                True, a deep copy will be made which will also copy the underlying
                data.

        Raises:
            KeyError: If a dataset is not found.

        Returns:
            `Datasets`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = self.copy()

        output = self.process_datasets(selection, names, operation="remove")

        if copy:
            return output.copy(deep=True)
        return output

    @staticmethod
    def __remove_variable(selection, dataset, cube_name, variable_names=None):
        """Remove a dataset's cube in-place.

        Selection is carried out exclusively using strings.

        The `variable_names` list is also modified in-place, so it needs to be
        mutable!

        """
        if not isinstance(dataset, str):
            dataset = dataset[0]
        if not isinstance(cube_name, str):
            cube_name = cube_name[0]
        assert isinstance(dataset, str)
        assert isinstance(cube_name, str)

        logger.debug("Removing variable: '{}'.".format(cube_name))

        if variable_names is None:
            variable_names = list(selection[dataset].variable_names("raw"))

        index = variable_names.index(cube_name)
        logger.debug("Removing cube from CubeList at index: {}.".format(index))
        del selection[dataset].cubes[index]
        logger.debug("Removing entry from names at index: {}.".format(index))
        del variable_names[index]

        logger.debug("Finished removing variable: '{}'.".format(cube_name))

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

        # Reformat the input dictionary for uniform indexing using dataset name tuples.
        formatted_values = []
        for key, value in search_dict.items():
            formatted_values.append((self.get_index(key, full_index=True), value))
        search_dict = dict(formatted_values)

        # Build a mutable representation of the current contents.
        contents = list(map(list, self.get("all", "all").items()))
        for i in range(len(contents)):
            contents[i][1] = list(contents[i][1])
        contents = dict(contents)
        removal_dict = contents.copy()

        up_to_date_raw = contents.copy()
        for search_dataset in up_to_date_raw:
            up_to_date_raw[search_dataset] = [
                name[0] for name in up_to_date_raw[search_dataset]
            ]

        # Look for a match for each desired variable.
        for search_dataset, search_variable in (
            (search_dataset, search_variable)
            for search_dataset, search_variables in search_dict.items()
            for search_variable in search_variables
        ):
            logger.debug("{} '{}: {}'.".format(which, search_dataset, search_variable))
            matches = 0
            for stored_variable in (
                stored_variable for stored_variable in contents[search_dataset]
            ):
                logger.debug(
                    "Checking '{}: {}'.".format(search_dataset, stored_variable)
                )
                if which == "remove" and contains(
                    stored_variable, search_variable, exact=exact
                ):
                    logger.debug(
                        "Removing '{}: {}'.".format(search_dataset, stored_variable)
                    )
                    self.__remove_variable(
                        selection,
                        search_dataset,
                        stored_variable,
                        variable_names=up_to_date_raw[search_dataset],
                    )
                    # Move on to the next variable.
                    logger.debug("Searching for next variable.")
                    break
                if which == "add":
                    if contains(stored_variable, search_variable, exact=exact):
                        logger.debug(
                            "Adding '{}: {}'.".format(search_dataset, stored_variable)
                        )
                        matches += 1
                        # We want to keep this variable, so we remove it from the
                        # dictionary which specifies which variables to remove.
                        removal_dict[search_dataset].remove(stored_variable)
            else:
                if which == "remove" or which == "add" and not matches:
                    raise KeyError(
                        "Variable '{}' not found for dataset '{}'.".format(
                            search_variable, search_dataset
                        )
                    )
        if which == "add":
            return self.dict_process_variables(
                selection, removal_dict, "remove", exact=exact
            )
        if which == "remove":
            # Remove empty datasets.
            selection.datasets[:] = [
                dataset for dataset in selection.datasets if dataset
            ]
        return selection

    def dict_select_variables(
        self, search_dict, exact=True, inplace=True, copy=False, strict=True
    ):
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
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection (see argument `copy`) before
                removing entries.
            copy (bool, optional): Only applies if `inplace` is False. If False
                (default), the new `Datasets` object will contain references to the
                same `Dataset` instances and associated cubes as the original. If
                True, a deep copy will be made which will also copy the underlying
                data.
            strict (bool, optional): If True (default) expect to select as many
                variables as given in `search_dict`.

        Raises:
            KeyError: If a key or value in `search_dict` is not found in the
                database.
            ValueError: If strict and if the number of output variables does not match
                the number of input variables.

        Returns:
            `Selection`: A copy of a subset of the original selection.

        """
        n_target_variables = sum(len(variables) for variables in search_dict.values())
        if inplace:
            selection = self
        else:
            selection = self.copy()

        output = self.dict_process_variables(
            selection, search_dict, which="add", exact=exact
        )

        if strict:
            # Count output cubes.
            n_output_variables = sum(len(dataset) for dataset in output)
            if n_target_variables != n_output_variables:
                raise ValueError(
                    "Expected to output {} variables, but got {}.".format(
                        n_target_variables, n_output_variables
                    )
                )
        if copy:
            return output.copy(deep=True)
        return output

    def dict_remove_variables(self, search_dict, exact=True, inplace=True, copy=False):
        """Remove variables matching `search_dict`.

        Cube pruning is performed after removal of entries to remove redundant cubes
        from dataset instances.

        Args:
            search_dict (dict): A dict with form
                  {dataset_name: (variable_name, ...), ...}, where `dataset_name` and
                  `variable_name` are of type `str`. This is used to select the
                  variables to process.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection (see argument `copy`) before
                removing entries.
            copy (bool, optional): Only applies if `inplace` is False. If False
                (default), the new `Datasets` object will contain references to the
                same `Dataset` instances and associated cubes as the original. If
                True, a deep copy will be made which will also copy the underlying
                data.

        Raises:
            KeyError: If a key or value in `search_dict` is not found in the
                database.

        Returns:
            `Datasets`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = self.copy()

        output = self.dict_process_variables(
            selection, search_dict, "remove", exact=exact
        )

        if copy:
            return output.copy(deep=True)
        return output

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
        logger.debug("Checking for duplicates.")
        if isinstance(names, str):
            names = (names,)

        # Build a mutable representation of the current contents.
        contents = list(map(list, self.get("all", "all").items()))
        for i in range(len(contents)):
            contents[i][1] = list(contents[i][1])
        contents = dict(contents)

        unpacked_datasets_variables = tuple(
            (dataset[0], variable_name)
            for dataset, variables in contents.items()
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
                f"Raw and pretty variable names are not unique ({n_duplicates} "
                "duplicate(s)). Use dictionaries for selection, or remove some "
                f"of the datasets '{duplicated_datasets}'."
            )

        removal_dict = contents.copy()

        # Look for a match for each desired variable.
        logger.debug("Selecting variables: '{}'.".format(names))
        for search_variable in names:
            matches = 0
            for stored_dataset, stored_variable in (
                (stored_dataset, stored_variable)
                for stored_dataset, stored_variables in contents.items()
                for stored_variable in stored_variables
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
                        # We want to keep this variable, so we remove it from the
                        # dictionary which specifies which variables to remove.
                        removal_dict[stored_dataset].remove(stored_variable)
            else:
                if which == "remove" or which == "add" and not matches:
                    raise KeyError("Variable '{}' not found.".format(search_variable))

        if which == "add":
            return self.dict_process_variables(
                selection, removal_dict, "remove", exact=exact
            )
        if which == "remove":
            # Remove empty datasets.
            selection.datasets[:] = [
                dataset for dataset in selection.datasets if dataset
            ]
        return selection

    def select_variables(
        self, names, exact=True, inplace=True, copy=False, strict=True
    ):
        """Return a new `Datasets` containing only variables matching `criteria`.

        Args:
            names (str or iterable): A name or series of names (variable_name, ...),
                for which a match with either raw or pretty names is sufficient. This
                is used to determine which variables to process.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection (see argument `copy`) before
                removing entries.
            copy (bool, optional): Only applies if `inplace` is False. If False
                (default), the new `Datasets` object will contain references to the
                same `Dataset` instances and associated cubes as the original. If
                True, a deep copy will be made which will also copy the underlying
                data.
            strict (bool, optional): If True (default) expect to select as many
                variables as given in `search_dict`.

        Raises:
            ValueError: If raw and pretty names are not unique across all stored
                variable names (raw and pretty) for all datasets. In this case, select
                variables using a dictionary instead of an iterable.
            KeyError: If a variable is not found in the database.
            ValueError: If strict and if the number of output variables does not match
                the number of input variables.

        Returns:
            `Datasets`: A copy of a subset of the original selection.

        """
        if isinstance(names, str):
            names = (names,)
        n_target_variables = len(names)
        if inplace:
            selection = self
        else:
            selection = self.copy()

        output = self.process_variables(selection, names, "add", exact=exact)

        if strict:
            # Count output cubes.
            n_output_variables = sum(len(dataset) for dataset in output)
            if n_target_variables != n_output_variables:
                raise ValueError(
                    "Expected to output {} variables, but got {}.".format(
                        n_target_variables, n_output_variables
                    )
                )
        if copy:
            return output.copy(deep=True)
        return output

    def remove_variables(self, names, exact=True, inplace=True, copy=False):
        """Return a new `Datasets` without the variables matching `criteria`.

        Cube pruning is performed after removal of entries to remove redundant cubes
        from dataset instances.

        Args:
            names (str or iterable): A name or series of names (variable_name, ...),
                for which a match with either raw or pretty names is sufficient. This
                is used to determine which variables to process.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.
            inplace (bool, optional): If True (default) modify the selection in-place
                without creating a copy. The returned selection will be the same
                selection as the original selection, but without the removed entries.
                If False, make a copy of the selection (see argument `copy`) before
                removing entries.
            copy (bool, optional): Only applies if `inplace` is False. If False
                (default), the new `Datasets` object will contain references to the
                same `Dataset` instances and associated cubes as the original. If
                True, a deep copy will be made which will also copy the underlying
                data.

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
            selection = self.copy()

        output = self.process_variables(selection, names, "remove", exact=exact)

        if copy:
            return output.copy(deep=True)
        return output

    def fill(self, *masks, reference_variable="burned.*area"):
        """Fill data gaps in-place.

        Args:
            *masks: Iterable of masks (each of type numpy.ndarray). These masks
                (boolean arrays) will be logical ORed with each other as well as the
                mask derived from `reference_variable`. The resulting mask will be
                applied to all variables in the dataset collection. If this mask
                completely masks out all missing data points, then no filling will be
                done. Common masks would be a land mask and a latitude mask.
            reference_variable (str or None): See description for `land_mask`. If str,
                use this to search for a variable whose mask will be used. If None, do
                not use a mask from a reference variable.

        """
        # Getting the shape does not realise lazy data.
        cube_shape = self.cubes[0].shape

        # Generate a final combined mask.
        if reference_variable is None:
            final_masks = []
        else:
            reference_cube = self.select_variables(
                reference_variable, exact=False, strict=True, inplace=False, copy=False
            ).cubes[0]
            final_masks = [reference_cube.data.mask]
        for mask in masks:
            assert len(mask.shape) in (2, 3)

        # Make sure masks have a matching shape.
        for mask in masks:
            if mask is not None:
                mask = match_shape(mask, cube_shape)
                final_masks.append(mask)

        if final_masks:
            combined_mask = reduce(np.logical_or, final_masks)
        else:
            combined_mask = np.zeros(cube_shape, dtype=np.bool_)

        # Process cubes on-by-one, while creating one-variable `Dataset` instances to
        # maintain flexible caching.
        # Use the generated mask in the 'filling' process.
        # TODO: Verify that this does indeed change the cubes in-place as expected!
        prev_shape = cube_shape
        for dataset in self:
            new_cubes = iris.cube.CubeList()
            for cube_slice in dataset.single_cube_slices():
                single_dataset = dataset[cube_slice]
                assert (
                    single_dataset.cube.shape == prev_shape
                ), "All cubes should have the same shape."
                prev_shape = single_dataset.cube.shape
                new_cubes.extend(fill_dataset(single_dataset, combined_mask))
            # TODO: Such an explicit assignment seems necessary to propagate the new cubes,
            # but the function could be optimised to remove old cubes in the process
            # to save a bit more memory. Ideally, it would be possible to change cubes
            # in-place (even while retrieving cached results, which complicates things
            # further, as the caching-decorator decorated function does not actually
            # get run)!
            dataset.cubes = new_cubes
        return self

    def homogenise_masks(self):
        for dataset in self.datasets:
            for i in range(len(dataset.cubes)):
                dataset.cubes[i] = homogenise_cube_mask(dataset.cubes[i])

    def show(self, variable_format="all"):
        """Print out a representation of the selection."""
        pprint(self.get(dataset_name="all", variable_format=variable_format))


def get_all_datasets(
    pretty_dataset_names=None, pretty_variable_names=None, ignore_names=IGNORED_DATASETS
):
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
    if ignore_names is None:
        ignore_names = []
    selection = Datasets()
    dataset_names = dir(wildfire_datasets)
    for dataset_name in [name for name in dataset_names if name not in ignore_names]:
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


def print_datasets_dates(selection):
    min_time, max_time, times_df = dataset_times(selection.datasets)
    print(times_df)


def prepare_selection(selection, *, min_time=None, max_time=None, which="all"):
    """Prepare cubes matching the given selection for analysis.

    Args:
        selection (`Datasets`): Selection specifying the variables to use.
        which (str): Controls which temporal processing is carried out. See return
            value documentation.

    Returns:
        If `which` is 'all', returns three `Datasets` (monthly, mean, climatology)
        which are calculated (and cached) at the same time to avoid repeat loading. If
        'monthly', 'mean' or 'climatology', return the first, second or third entry
        respectively.

    Raises:
        ValueError: If the time selection limits are not defined using the `min_time`
            and `max_time` parameters or the datasets.

    """
    if any(given_time is None for given_time in (min_time, max_time)):
        data_min_time, data_max_time, times_df = dataset_times(selection.datasets)
        if any(data_time is None for data_time in (data_min_time, data_max_time)):
            assert all(
                result is None for result in (data_min_time, data_max_time, times_df)
            )
        else:
            if min_time is None:
                min_time = data_min_time
            if max_time is None:
                max_time = data_max_time

    if any(selection_time is None for selection_time in (min_time, max_time)):
        raise ValueError("Time selection limits are undefined.")

    if which == "all":
        dataset_function = get_monthly_mean_climatology
        result_collection = (Datasets(), Datasets(), Datasets())
    elif which == "mean":
        dataset_function = get_mean
        result_collection = (Datasets(),)
    elif which == "climatology":
        dataset_function = get_climatology
        result_collection = (Datasets(),)
    elif which == "monthly":
        dataset_function = get_monthly
        result_collection = (Datasets(),)
    else:
        raise ValueError(f"Unknown value for which '{which}'")

    logger.info(f"Preparing selection '{which}' between '{min_time}' and '{max_time}'")

    # Use get_ncpus() even even for a ThreadPool since large portions of the code
    # release the GIL.
    outputs = Parallel(n_jobs=get_ncpus(), prefer="threads")(
        delayed(partial(dataset_function, min_time=min_time, max_time=max_time))(
            dataset
        )
        for dataset in selection
    )
    for result_datasets, output in zip(result_collection, zip(*outputs)):
        result_datasets += output

    if len(result_collection) == 1:
        result_collection = result_collection[0]
    return result_collection


if __name__ == "__main__":
    # LOGGING["handlers"]["console"]["level"] = "DEBUG"
    logging.config.dictConfig(LOGGING)

    # from wildfires.data.datasets import (
    #     AvitabileThurnerAGB,
    #     Copernicus_SWI,
    #     iris_memory,
    #     ERA5_DryDayPeriod,
    #     ERA5_CAPEPrecip,
    #     MOD15A2H_LAI_fPAR,
    # )
    # from dateutil.relativedelta import relativedelta
    # selection = Datasets() + MOD15A2H_LAI_fPAR()

    selection = get_all_datasets(
        ignore_names=(
            "AvitabileAGB",
            "CRU",
            "ESA_CCI_Fire",
            "ESA_CCI_Landcover",
            "ESA_CCI_Soilmoisture",
            "ESA_CCI_Soilmoisture_Daily",
            "GFEDv4s",
            "GPW_v4_pop_dens",
            "GSMaP_dry_day_period",
            "LIS_OTD_lightning_time_series",
            "Simard_canopyheight",
            "Thurner_AGB",
        )
    )

    selected_names = [
        "AGBtree",
        "maximum temperature",
        "minimum temperature",
        "Soil Water Index with T=1",
        "ShrubAll",
        "TreeAll",
        "pftBare",
        "pftCrop",
        "pftHerb",
        "monthly burned area",
        "dry_days",
        "dry_day_period",
        "precip",
        "SIF",
        "popd",
        "Combined Flash Rate Monthly Climatology",
        "VODorig",
        "Fraction of Absorbed Photosynthetically Active Radiation",
        "Leaf Area Index",
    ]

    selection = selection.select_variables(selected_names, strict=True)

    # selection = Datasets() + AvitabileThurnerAGB() + Copernicus_SWI()
    # selection.select_variables(
    #     ("AGBtree", "Soil Water Index with T=5", "Soil Water Index with T=20")
    # )

    # The `min_time` and `max_time` determined here would be used automatically!
    min_time, max_time, times_df = dataset_times(selection.datasets)

    selection.show("pretty")

    max_time = max_time
    # max_time = min_time + relativedelta(years=+1)
    output_datasets = prepare_selection(selection, min_time=min_time, max_time=max_time)

    # The following function calls should (in theory) only retrieve data previously
    # cached during the execution of the above function call.
    monthly_datasets = prepare_selection(
        selection, min_time=min_time, max_time=max_time, which="monthly"
    )
    climatology_datasets = prepare_selection(
        selection, min_time=min_time, max_time=max_time, which="climatology"
    )
    mean_datasets = prepare_selection(
        selection, min_time=min_time, max_time=max_time, which="mean"
    )
