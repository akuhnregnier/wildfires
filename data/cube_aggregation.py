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
import os
import pickle
import re
from collections import OrderedDict, defaultdict, namedtuple
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
memory = Memory(location=DATA_DIR, verbose=100)

TARGET_PICKLES = tuple(
    os.path.join(DATA_DIR, filename)
    for filename in (
        "selected_cubes.pickle",
        "mean_cubes.pickle",
        "monthly_climatologies.pickle",
    )
)

# These classes need to be defined here (global) in order for pickling of Selection
# instances to work.
Dataset = namedtuple("Dataset", ("raw", "pretty"))
Variable = namedtuple("Variable", ("raw", "pretty"))


def return_none():
    return


def contains(entry, item, exact=True, str_only=False):
    """String matching for selected fields.

    Args:
        entry (namedtuple): Tuple wherein existence of `item` is checked.
        item: Item to look for.
        exact (bool): If True, only accept exact matches for strings,
            ie. replace `item` with `^item$` before using `re.search`.
        str_only (bool): If True, only compare strings and ignore other items in
            `entry`.

    """
    for field in entry._fields:
        value = getattr(entry, field)
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


class Selection:
    """Keep track of datasets and associated variables.

    Both original (raw) and user-friendly (pretty) versions of variable names are
    recorded. Every dataset's raw and pretty variable names are guaranteed to be
    unique amongst all raw and pretty variable names in that dataset (this is not
    guaranteed amongst all datasets!).

    A raw and pretty name is also stored for each dataset. These names are guaranteed
    to be unique amongst all raw and pretty dataset names in the selection.

    Examples:
        >>> sel = Selection()
        >>> sel = sel.add("HYDE", ("popd", "pop density"))
        >>> sel = sel.add("GFEDv4", ("monthly burned area", "burned area"))
        >>> sel = sel.add("Dataset", "raw_name")
        >>> all_all = sel.get(dataset="all", variable_format="all")
        >>> all_all == {
        ...     ("HYDE", "HYDE"): (("popd", "pop density"),),
        ...     ("GFEDv4", "GFEDv4"): (("monthly burned area", "burned area"),),
        ...     ("Dataset", "Dataset"): (("raw_name", "raw_name"),),
        ... }
        True
        >>> # Test a single dataset.
        >>> sel.get(dataset="HYDE", variable_format="pretty")
        {'HYDE': ('pop density',)}
        >>> # Duplicated variable names within a dataset are not allowed!
        >>> try:
        ...     sel.add("HYDE", "popd")
        ... except ValueError as exception:
        ...     exception.args[0] == (
        ...         "raw variable name 'popd' is already present in {Dataset(raw='HYDE', "
        ...         "pretty='HYDE'): [Variable(raw='popd', "
        ...         "pretty='pop density')]}"
        ...     )
        True
        >>> long_sel = Selection()
        >>> long_sel = long_sel.add("HYDE", ("popd", "pop density"))
        >>> long_sel = long_sel.add("HYDE", ("grazing land area", "grazing area"))
        >>> long_sel = long_sel.add("GFEDv4", ("monthly burned area", "burned area"))
        >>> long_sel = long_sel.add(
        ...     "GFEDv4", ("monthly average co2 emissions", "co2 emissions")
        ... )
        >>> long_sel = long_sel.add("Dataset", "raw_name")
        >>> # Variable selection can be done either with dictionaries of tuples
        >>> long_sel.dict_select_variables(
        ...         {"HYDE": ("popd", "grazing land area"), "Dataset": ("raw_name",)}
        ...     ) == Selection().add("HYDE", ("popd", "pop density")).add(
        ...         "HYDE", ("grazing land area", "grazing area")
        ...     ).add(
        ...         "Dataset", "raw_name"
        ...     )
        True
        >>> # or only with tuple of variable names.
        >>> sel.select_variables(("burned area", "popd", "raw_name")) == sel
        True
        >>> # Dataset instances can also be incorporated.
        >>> from .datasets import HYDE
        >>> from ..tests.test_datasets import data_is_available
        >>> instance_sel = Selection()
        >>> if data_is_available:
        ...     assert Selection().add(("HYDE", "HYDE", HYDE()), "popd").instances

    """

    def __init__(self, dataset_variables=None):
        # Set up the cache (assign empty dicts to cache-related variables).
        self.clear_cache()

        self.dataset_variables = defaultdict(list)
        self.__instances = defaultdict(return_none)
        if dataset_variables:
            self._add_dict(dataset_variables)

    def __eq__(self, other):
        if isinstance(other, Selection):
            return self.immutable == other.immutable
        return NotImplemented

    def __hash__(self):
        return hash(self.immutable)

    def __add__(self, other):
        if isinstance(other, Selection):
            new_entries = other.dataset_variables
        elif isinstance(other, dict):
            new_entries = other

        return deepcopy(self)._add_dict(new_entries)

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Selection):
            new_entries = other.dataset_variables
        elif isinstance(other, dict):
            new_entries = other
        else:
            return NotImplemented

        return self._add_dict(new_entries)

    def __str__(self):
        return pformat(self.get("all", "pretty"))

    def __repr__(self):
        return pformat(self.immutable)

    def copy(self):
        return deepcopy(self)

    def _add_dict(self, dataset_variables):
        for dataset, variables in dataset_variables.items():
            for variable in variables:
                self.add(dataset, variable)
        return self

    def clear_cache(self):
        logger.debug("Resetting cache to initial state (id '{}').".format(id(self)))
        self.__formatted = defaultdict(OrderedDict)
        self.__last_state = defaultdict(tuple)
        return self

    @property
    def raw_variable_names(self):
        """Return a tuple of all raw variable names."""
        var_names = []
        for names in self.get("all", "raw").values():
            var_names.extend(names)
        return tuple(var_names)

    @property
    def pretty_variable_names(self):
        """Return a tuple of all pretty variable names."""
        var_names = []
        for names in self.get("all", "pretty").values():
            var_names.extend(names)
        return tuple(var_names)

    @property
    def raw_dataset_names(self):
        """Return a tuple of all raw dataset names."""
        return tuple(self.get("all", "raw"))

    @property
    def pretty_dataset_names(self):
        """Return a tuple of all pretty dataset names."""
        return tuple(self.get("all", "pretty"))

    @property
    def instances(self):
        """Return a tuple of dataset instances."""
        return tuple(self.__instances[dataset] for dataset in self.get("all", "all"))

    @property
    def cubes(self):
        """Return all cubes."""
        self.prune_cubes()
        all_cubes = iris.cube.CubeList()
        for dataset in self.dataset_variables:
            all_cubes.extend(self.__instances[dataset].cubes)
        return all_cubes

    @property
    def immutable(self):
        """Retrieve underlying data using tuples of variable names instead of lists."""
        immutable_repr = dict()
        for dataset, variables in self.dataset_variables.items():
            immutable_repr[dataset] = tuple(sorted(variables))
        return tuple(sorted(immutable_repr.items()))

    @staticmethod
    def pack_input(var, single_type=str, elements=2, fill_source=0):
        """Return a filled tuple with `elements` items.

        Args:
            var (iterable of `single_type` or `single_type`): Input variable which
                will be transformed.
            single_type (class or tuple of class): Atomic type(s) that will be treated
                as single items.
            elements (int): Number of elements in the final tuple.
            fill_source (int, None, or iterable of int or None): Determines how to pad
                input such that it contains `elements` items. No existing items in
                `var will be altered. If `fill_source` is an int or None, it is
                treated as an iterable containing `fill_source` (`elements` -
                len(`var`)) times, where len(`var`) refers to the number of
                `single_type` items supplied in `var` (which may be only one, in which
                case `var` is internally transformed to be a 1-element iterable
                containing `var`). The `fill_source` iterable must contain at least
                (`elements` - len(`var`)) items, since this is the number of slots
                that need to be filled in order for the output to contain `elements`
                items. If `fill_source[-i]` is an int, `output[-i]` will be inserted into
                `output` at index `elements - i`. If `fill_source[-i]` is None, None
                will be inserted. Surplus `fill_source` items will be trimmed starting
                from the left (thus the -i index notation above).
        Raises:
            ValueError: If `var` is an iterable of `single_type` and contains more than
                `elements` items.
            TypeError: If `var` is an iterable and its items are not all of type
                `single_type`.
            TypeError: If `fill_source` contains types other than int and NoneType.
            IndexError: If `len(fill_source)` < (`elements` - len(`var`)).
            IndexError: If `fill_source[-i]` is an int and
                `fill_source[-i]` >= `elements` - i.

        Returns:
            tuple: tuple with `elements` items.

        Examples:
            >>> pack_input = Selection.pack_input
            >>> pack_input("testing")
            ('testing', 'testing')
            >>> pack_input(("foo",))
            ('foo', 'foo')
            >>> pack_input(("foo", "bar"), elements=3, fill_source=1)
            ('foo', 'bar', 'bar')
            >>> pack_input("foo", elements=2, fill_source=None)
            ('foo', None)
            >>> pack_input("foo", elements=3, fill_source=(0, None))
            ('foo', 'foo', None)
            >>> # Surplus `fill_source` items will be trimmed starting from the left.
            >>> pack_input("foo", elements=3, fill_source=(99, 0, None))
            ('foo', 'foo', None)
            >>> pack_input(("foo", "bar"), elements=5, fill_source=(1, 2, None))
            ('foo', 'bar', 'bar', 'bar', None)

        """
        if not isinstance(var, single_type):
            if not all(isinstance(single_var, single_type) for single_var in var):
                raise TypeError(
                    "Expected items to be of type(s) '{}', but got types '{}'.".format(
                        single_type, [type(single_var) for single_var in var]
                    )
                )
            if len(var) > elements:
                raise ValueError(
                    "Expected at most {} item(s), got {}.".format(elements, len(var))
                )
            if len(var) == elements:
                return tuple(var)
            # Guarantee that `var` is a list, and make a copy so the input is not
            # changed unintentionally.
            var = list(var)
        else:
            var = [var]

        fill_source_types = (int, type(None))
        if not isinstance(fill_source, fill_source_types):
            if not all(
                isinstance(single_source, fill_source_types)
                for single_source in fill_source
            ):
                raise TypeError(
                    "Expected fill_source to be of types '{}', but got types '{}'.".format(
                        fill_source_types,
                        [type(single_source) for single_source in fill_source],
                    )
                )
            # Again, make a copy.
            fill_source = fill_source[:]
        else:
            fill_source = [fill_source] * (elements - len(var))

        n_missing = elements - len(var)
        for i in range(-n_missing, 0):
            if fill_source[i] is None:
                fill_value = None
            else:
                fill_value = var[fill_source[i]]
            var.append(fill_value)
        return tuple(var)

    def get_index(self, dataset):
        """Get dataset index tuple from a dataset string.

        The given name may match either the raw or the pretty name.

        """
        for dataset_name_tuple in self.dataset_variables:
            if dataset in dataset_name_tuple:
                return dataset_name_tuple

    def add(self, dataset, variable):
        """Add a dataset name(s): variable name(s) pair to the database.

        Note:
            - If only one name is given for `dataset` or `variable`, the
              first (raw) name is used for both the raw and pretty names.
            - Specifying a dataset with the same name(s) but different instances in
              any order will raise a ValueError to enforce consistent use of
              instances.

        Args:
            dataset (str, iterable, or dict): A raw name (str), a raw and pretty name
                (iterable of str), or a dictionary with the mandatory key "raw" (to
                specify the raw name (str)) and the optional keys "pretty" (to specify
                the pretty name (str)) and "instance" (
                wildfires.data.datasets.Dataset or None). Each raw name, pretty name
                and instance (unless it is None) must not already be present.
            variable (str, iterable, or dict): Same as above without the "instance"
                key so at most 2 names can be given.

        Raises:
            ValueError: If the provided dataset names match multiple existing
                datasets (partially). Matches are made using all combinations of raw
                and pretty names.
            ValueError: If the provided dataset matches a single existing dataset only
                partially, not wholly (eg. only their raw names, but not their pretty
                names match).
            ValueError: If either the given raw or pretty variable name matches a
                any of the existing raw and pretty variable names for the given
                dataset.

        """
        if isinstance(dataset, dict):
            new_instance = dataset.get("instance", None)
            dataset = Dataset(
                raw=dataset.get("raw"), pretty=dataset.get("pretty", dataset.get("raw"))
            )
        else:
            # Handles cases (str, str), (str,), or (str, str, Dataset). Cases like
            # (str, Dataset) won't fail, but will produce unexpected results.
            raw, pretty, new_instance = self.pack_input(
                dataset,
                # Use type(None) as a placeholder for missing dataset instances.
                single_type=(str, wildfire_datasets.Dataset, type(None)),
                elements=3,
                fill_source=(0, None),
            )
            dataset = Dataset(raw, pretty)
        if isinstance(variable, dict):
            variable = Variable(
                raw=variable.get("raw"),
                pretty=variable.get("pretty", variable.get("raw")),
            )
        else:
            variable = Variable(
                *self.pack_input(variable, single_type=str, elements=2, fill_source=0)
            )

        dataset_match_count = 0
        prev_match_key = None
        for stored_dataset in self.dataset_variables:
            increment = sum(
                (
                    dataset.raw == stored_dataset.raw,
                    dataset.pretty == stored_dataset.pretty,
                )
            )
            if increment:
                if prev_match_key is not None:
                    raise ValueError(
                        (
                            "A match between the new dataset '{}' and "
                            "both the existing datasets '{}' and '{}' was found."
                        ).format(dataset, prev_match_key, stored_dataset)
                    )

                # If there was a name match, make sure that the instances match too.
                increment += new_instance == self.__instances[stored_dataset]

                prev_match_key = stored_dataset
                dataset_match_count += increment

        # Add 1 here to account for the fact that the instance is not stored as part
        # of the namedtuple, but contributes to the dataset_match_count nonetheless.
        if dataset_match_count not in {0, len(dataset._fields) + 1}:
            raise ValueError(
                (
                    "Expected either no matches or a complete match, but matched "
                    "{} field(s). The dataset '{}' is already partially present."
                ).format(dataset_match_count, dataset)
            )

        for key, var_name in zip(variable._fields, variable):
            existing = [
                getattr(stored_var_names, key)
                for stored_var_names in self.dataset_variables[dataset]
            ]
            if var_name in existing:
                raise ValueError(
                    "{} variable name '{}' is already present in {{{}: {}}}".format(
                        key, var_name, dataset, self.dataset_variables[dataset]
                    )
                )
        logger.debug("Adding variable '{}' for dataset '{}'.".format(variable, dataset))
        self.dataset_variables[dataset].append(variable)
        logger.debug(
            "Adding instance '{}' for dataset '{}'.".format(new_instance, dataset)
        )
        self.__instances[dataset] = new_instance
        return self

    def __return_cached_repr(self, dataset, variable_format):
        """Return cached representation.

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
        if dataset == "all":
            return self.__formatted[variable_format]
        if variable_format == "all":
            selected_key = self.get_index(dataset)
        else:
            selected_key = getattr(self.get_index(dataset), variable_format)
        return {selected_key: self.__formatted[variable_format][selected_key]}

    def get(self, dataset="all", variable_format="raw"):
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
        logger.debug("Get() called with Selection '{}'".format(id(self)))
        if self.__last_state[variable_format] == self.immutable:
            # Return previously calculated dictionary.
            logger.debug(
                "Using cached representation for params: {}, {}.".format(
                    dataset, variable_format
                )
            )
            return self.__return_cached_repr(dataset, variable_format)

        # Calculate the correct representation for the current contents of
        # self.dataset_variables. Do this for all datasets, regardless of which
        # datasets are actually requested. If `self.dataset_variables` does not
        # change from one call to the next, the cached calculations below will be
        # retrieved for other requested datasets.
        logger.debug(
            "Updating representation for variable_format: {}.".format(variable_format)
        )
        if variable_format == "all":
            self.__formatted[variable_format] = OrderedDict(self.immutable)
        else:
            self.__formatted[variable_format] = OrderedDict(
                (
                    getattr(stored_dataset, variable_format),
                    tuple(
                        getattr(variable, variable_format)
                        for variable in stored_variables
                    ),
                )
                for stored_dataset, stored_variables in OrderedDict(
                    self.immutable
                ).items()
            )

        # Record that the dictionary now stored in
        # self.__formatted[variable_format] reflects the current database status.
        self.__last_state[variable_format] = self.immutable

        logger.debug(
            "Returning newly calculated representation for dataset: {}.".format(dataset)
        )
        return self.__return_cached_repr(dataset, variable_format)

    def prune_cubes(self, clear_cache=True):
        """Remove cubes from Dataset instance which do not have a variable name entry.

        Also remove datasets containing no variables.
        Datasets with an instance `None` are ignored.

        Args:
            clear_cache (bool): If True (default), clear the cache after each
                non-trivial operation. This is to avoid lingering references to
                instances.

        Raises:
            ValueError: If `self` has records of variables unknown to the Dataset
                instance.

        """
        if self.__last_state["prune"] == self.immutable:
            logger.debug("No change in dataset_variables since last pruning.")
            return

        logger.debug("Pruning dataset_variables.")
        empty_datasets = []
        for dataset, variables in self.dataset_variables.items():
            logger.debug("Pruning dataset: '{}'.".format(dataset))
            if not variables:
                empty_datasets.append(dataset)
                continue
            if self.__instances[dataset] is None:
                continue
            raw_instance_names = set(
                cube.name() for cube in self.__instances[dataset].cubes
            )
            raw_stored_names = set(variable.raw for variable in variables)
            unknown_variables = raw_stored_names - raw_instance_names
            if unknown_variables:
                raise ValueError(
                    "The variables '{}' were not found in the instance '{}'".format(
                        unknown_variables, self.__instances[dataset]
                    )
                )

            logger.debug(
                "Number of stored variables: {}.".format(len(raw_stored_names))
            )

            logger.debug(
                "Number of dataset instance variables: {}.".format(
                    len(raw_instance_names)
                )
            )

            def selection_func(cube):
                return cube.name() in raw_stored_names

            logger.debug(
                "Number of cubes before: {}.".format(
                    len(self.__instances[dataset].cubes)
                )
            )
            # Keep only the cubes referred to by the stored variable names.
            self.__instances[dataset].cubes = self.__instances[dataset].cubes.extract(
                iris.Constraint(cube_func=selection_func)
            )
            logger.debug(
                "Number of cubes after: {}.".format(
                    len(self.__instances[dataset].cubes)
                )
            )

        for empty_dataset in empty_datasets:
            logger.debug("Removing empty dataset {}.".format(empty_dataset))
            del self.dataset_variables[empty_dataset]

        # FIXME: Isn't this done automatically if necessary?
        if clear_cache:
            self.clear_cache()
        self.__last_state["prune"] = self.immutable
        return self

    def process_datasets(self, selection, names, operation="select"):
        """Process datasets in `selection` which match `names`.

        Args:
            selection (`Selection`): Selection containing datasets to process.
            names (str or iterable): A name or series of names (dataset_name, ...),
                for which a match with either raw or pretty names is sufficient.
            operation (str): If "select", add matching datasets to `selection`. If
                "remove", remove matching datasets from `selection`.

        Raises:
            KeyError: If a dataset is not found in the database.

        Returns:
            `Selection`: `selection` processed using `processing_func`.
        """
        if isinstance(names, str):
            names = (names,)
        for search_dataset in names:
            for stored_dataset in self.dataset_variables:
                if search_dataset in stored_dataset:
                    if operation == "select":
                        selection._add_dict(
                            {stored_dataset: self.dataset_variables[stored_dataset]}
                        )
                    elif operation == "remove":
                        del selection.dataset_variables[stored_dataset]
                    else:
                        raise ValueError("Invalid operation '{}'".format(operation))
                    break
            else:
                raise KeyError("Dataset '{}' not found.".format(search_dataset))
        return selection

    def select_datasets(self, names):
        """Return a new `Selection` containing only datasets matching `names`.

        Args:
            names (str or iterable): A name or series of names (dataset_name, ...),
                for which a match with either raw or pretty names is sufficient.

        Raises:
            KeyError: If a dataset is not found in the database.

        Returns:
            `Selection`: A copy of a subset of the original selection.

        """
        return self.process_datasets(Selection(), names, operation="select")

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
            KeyError: If a dataset is not found in the database.

        Returns:
            `Selection`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = deepcopy(self)
        return self.process_datasets(selection, names, operation="remove")

    @staticmethod
    def __remove_entry(selection, dataset, variable):
        """Remove a dataset's variable entry in-place."""
        selection.dataset_variables[dataset].remove(variable)
        # FIXME: Add prune_cubes() here??

    def __add_entry(self, selection, dataset, variable):
        """Add a dataset's variable entry in-place."""
        selection.add(
            (dataset.raw, dataset.pretty, self.__instances[dataset]), variable
        )

    def dict_process_variables(
        self, selection, search_dict, processing_func, exact=True
    ):
        """Modify the selection's matching variable entries using the given function.

        Args:
            selection (`Selection`): Selection containing variables to process.
            search_dict (dict): A dict with form
                  {dataset_name: (variable_name, ...), ...}, where `dataset_name` and
                  `variable_name` are of type `str`. This is used to select the
                  variables to process.
            processing_func (`function`): Function which takes arguments
                (`Selection`, `Dataset`, `Variable`)
                and modifies the input selection in-place.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Returns:
            `selection` processed using `processing_func`.

        """
        # Look for a match for each desired variable.
        for search_dataset, search_variable in (
            (search_dataset, search_variable)
            for search_dataset, target_variables in search_dict.items()
            for search_variable in target_variables
        ):
            for stored_dataset, stored_variable in (
                (stored_datasets, stored_variable)
                for stored_datasets, stored_variables in self.get(
                    search_dataset, variable_format="all"
                ).items()
                for stored_variable in stored_variables
            ):
                if contains(stored_variable, search_variable, exact=exact):
                    # We found a match, process this.
                    processing_func(selection, stored_dataset, stored_variable)
                    # Move on to the next next desired variable.
                    break
            else:
                raise KeyError(
                    "Variable '{}' not found for dataset '{}'.".format(
                        search_variable, search_dataset
                    )
                )
        return selection

    def dict_select_variables(self, search_dict, exact=True):
        """Return a new `Selection` containing only variables matching `search_dict`.

        Args:
            search_dict (dict): A dict with form
                  {dataset_name: (variable_name, ...), ...}, where `dataset_name` and
                  `variable_name` are of type `str`. This is used to select the
                  variables to process.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Raises:
            KeyError: If a key or value in `search_dict` is not found in the
                database.

        Returns:
            `Selection`: A copy of a subset of the original selection.

        """
        selection = Selection()
        return self.dict_process_variables(
            selection, search_dict, self.__add_entry, exact=exact
        )

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
            `Selection`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = deepcopy(self)
        selection = self.dict_process_variables(
            selection, search_dict, self.__remove_entry, exact=exact
        )
        self.prune_cubes()
        return selection

    def process_variables(self, selection, names, processing_func, exact=True):
        """Modify the selection's matching variable entries using the given function.

        Args:
            selection (`Selection`): Selection containing variables to process.
            names (str or iterable): A name or series of names (variable_name, ...),
                for which a match with either raw or pretty names is sufficient. This
                is used to determine which variables to process.
            processing_func (`function`): Function which takes arguments
                (`Selection`, `Dataset`, `Variable`)
                and modifies the input selection in-place.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Raises:
            ValueError: If raw and pretty names are not unique across all stored
                variable names (raw and pretty) for all datasets. In this case, select
                variables using a dictionary instead of an iterable.
            KeyError: If a variable is not found in the database.

        Returns:
            `Selection`: `selection` processed using `processing_func`.

        """
        if isinstance(names, str):
            names = (names,)
        unpacked_datasets_variables = tuple(
            (dataset.raw, variable_name)
            for dataset, variables in self.dataset_variables.items()
            for variable in variables
            # Use set() here, since the pretty variable name could be the same as the
            # raw variable name if no pretty variable has been provided. This would
            # result in a false positive, as we are only interested in identical names
            # for different variables.
            for variable_name in set(variable)
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
            # Check against all stored datasets.
            for stored_dataset, stored_variable in (
                (stored_dataset_tuple, stored_variable)
                for stored_dataset_tuple, stored_variables in (
                    self.dataset_variables.items()
                )
                for stored_variable in stored_variables
            ):
                # Check against all variables in the dataset.
                if contains(stored_variable, search_variable, exact=exact):
                    # We found a match, process this.
                    logger.debug(
                        "Selecting '{}: {}'.".format(stored_dataset, stored_variable)
                    )
                    processing_func(selection, stored_dataset, stored_variable)
                    # Move on to the next next variable.
                    break
            else:
                raise KeyError("Variable '{}' not found.".format(search_variable))
        return selection

    def select_variables(self, names, exact=True):
        """Return a new `Selection` containing only variables matching `criteria`.

        Args:
            names (str or iterable): A name or series of names (variable_name, ...),
                for which a match with either raw or pretty names is sufficient. This
                is used to determine which variables to process.
            exact (bool, optional): If False, accept partial matches for variable
                names using `re.search`.

        Raises:
            ValueError: If raw and pretty names are not unique across all stored
                variable names (raw and pretty) for all datasets. In this case, select
                variables using a dictionary instead of an iterable.
            KeyError: If a variable is not found in the database.

        Returns:
            `Selection`: A copy of a subset of the original selection.

        """
        selection = Selection()
        return self.process_variables(selection, names, self.__add_entry, exact=exact)

    def remove_variables(self, names, inplace=True, exact=True):
        """Return a new `Selection` without the variables matching `criteria`.

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
            `Selection`: A (copy of a) subset of the original selection.

        """
        if inplace:
            selection = self
        else:
            selection = deepcopy(self)
        selection = self.process_variables(
            selection, names, self.__remove_entry, exact=exact
        )
        selection.prune_cubes()
        return selection

    def show(self, variable_format="all"):
        """Print out a representation of the selection."""
        pprint(self.get(dataset="all", variable_format=variable_format))


def get_all_datasets(pretty_dataset_names={}, pretty_variable_names={}):
    """Get all valid datasets defined in the `wildfires.data.datasets` module.

    Args:
        pretty_variable_names (dict): Dictionary mapping raw to pretty variable names
            ({raw: pretty, ...}).

    Returns:
        `Selection`: Selection object describing the datasets.

    """
    selection = Selection()
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
                instance = obj()
                for cube in instance.cubes:
                    selection.add(
                        (
                            dataset_name,
                            pretty_dataset_names.get(dataset_name, dataset_name),
                            instance,
                        ),
                        (
                            cube.name(),
                            pretty_variable_names.get(cube.name(), cube.name()),
                        ),
                    )
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
                value = self.__class__(value, self.__expansion_count).hashable()
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

    Do this by giving joblib a different version of the input arguments to cache,
    while still passing the normal arguments to the decorated function.

    """

    @wraps(func)
    def takes_original_selection(*args, **kwargs):
        """Function that is visible to the outside."""
        if not isinstance(args[0], Selection) or any(
            isinstance(arg, Selection)
            for arg in list(args[1:]) + list(kwargs.values()) + list(kwargs.keys())
        ):
            raise TypeError(
                "The first positional argument, and only the first argument "
                "should be a Selection instance."
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


@clever_cache
def prepare_selection(selection, verbose=True):
    """Prepare cubes matching the given selection for analysis.

    Args:
        selection (`Selection`): Selection specifying the variables to use.
        averaging (str or None): If None, do not perform any averaging. If "mean",
            average original monthly data over all time. If "monthly climatology",
            produce a monthly climatology.
        verbose (bool): If True, print output about the datasets, such as the validity
            periods.

    """
    min_time, max_time, times_df = dataset_times(selection.instances)
    if verbose:
        print(times_df)

    # Limit the amount of data that has to be processed.
    logger.info("Limiting data")
    for dataset in selection.instances:
        dataset.limit_months(min_time, max_time)
    logger.info("Finished limiting data")

    # Regrid cubes to the same lat-lon grid.
    # TODO: change lat and lon limits and also the number of points!!
    # Always work in 0.25 degree steps? From the same starting point?
    logger.info("Starting regridding of all datasets")
    for dataset in selection.instances:
        dataset.regrid()
    logger.info("Finished regridding of all datasets.")

    logger.info("Starting temporal upscaling.")
    # Join up all the cubes.
    cubes = iris.cube.CubeList()
    for dataset in selection.instances:
        cubes.extend(dataset.get_monthly_data(min_time, max_time))
    logger.info("Finished temporal upscaling.")

    # Get list of names for further selection.
    # pprint(selection.raw_variable_names)

    # Realise data - not necessary when using iris.save, but necessary here as pickle
    # files are being used!
    [c.data for c in cubes]
    with open(TARGET_PICKLES[0], "wb") as f:
        pickle.dump(cubes, f, protocol=-1)
    logger.info(cubes)

    logger.info("Calculating total mean.")
    mean_cubes = iris.cube.CubeList()
    for cube in tqdm(cubes):
        mean_cubes.append(cube.collapsed("time", iris.analysis.MEAN))
    [c.data for c in mean_cubes]
    with open(TARGET_PICKLES[1], "wb") as f:
        pickle.dump(mean_cubes, f, protocol=-1)
    logger.info(mean_cubes)

    logger.info("Calculating monthly climatology.")
    averaged_cubes = iris.cube.CubeList()
    for cube in tqdm(cubes):
        if not cube.coords("month_number"):
            iris.coord_categorisation.add_month_number(cube, "time")
        averaged_cubes.append(cube.aggregated_by("month_number", iris.analysis.MEAN))
    [c.data for c in averaged_cubes]
    with open(TARGET_PICKLES[2], "wb") as f:
        pickle.dump(averaged_cubes, f, protocol=-1)

    return cubes, mean_cubes, averaged_cubes


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
    aggregated_cubes = prepare_selection(selection)
