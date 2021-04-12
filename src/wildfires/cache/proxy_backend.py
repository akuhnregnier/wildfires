# -*- coding: utf-8 -*-
"""Proxy Joblib backend."""
# -*- coding: utf-8 -*-
import ast
import datetime
import logging
import os
import re
from functools import partial

import cloudpickle
from joblib import register_store_backend
from joblib._store_backends import CacheItemInfo, concurrency_safe_rename, mkdirp
from lazy_object_proxy.slots import Proxy

from .backend import CommonBackend
from .hashing import (
    Hasher,
    _default_context_managers,
    _default_guarded_hashers,
    _default_initial_hashers,
)

logger = logging.getLogger(__name__)
HASHES_ONLY = object()


class HashProxyHasher(Hasher):
    """Compute the hash with support for lazy proxied objects."""

    @staticmethod
    def test_argument(arg):
        return isinstance(arg, HashProxy)

    @staticmethod
    def hash(arg):
        return arg.hashed_value


class Factory:
    """Factory function that keeps track of whether it was called."""

    __slots__ = ("factory", "_was_called")

    def __init__(self, factory):
        self.factory = factory
        self._was_called = False

    @property
    def was_called(self):
        return self._was_called

    def __call__(self, *args, **kwargs):
        self._was_called = True
        return self.factory(*args, **kwargs)


class HashProxy(Proxy):
    """Lazy proxy containing a pre-calculated hash value."""

    __slots__ = ("_hash_value", "_hash_func")

    def __init__(self, factory, hash_func, hash_value=None):
        """Initialise the proxy with the factory function and pre-defined hash.

        Args:
            factory (Factory): Factory function that generates the object to be
                proxied on demand.
            hash_value (int): Pre-computed hash value. This will be invalidated once
                `factory` is called. If None is given, the hash value will be computed
                every time (which required calling `factory`).

        """
        if not isinstance(factory, Factory):
            raise TypeError("'factory' needs to be a Factory instance.")
        super().__init__(factory)
        self._hash_value = hash_value
        self._hash_func = hash_func

    @property
    def hashed_value(self):
        if self._hash_value is None or self.__factory__.was_called:
            self._hash_value = None  # Ensure this will never be accessed.
            return self._hash_func(self.__wrapped__)
        return self._hash_value


def cache_hash_value(obj, hash_func, func=None):
    """Cache the hash value of a given object.

    An optional function can be supplied (with signature func(obj) -> obj) which
    should not change the hash value of `obj`.

    """
    if func is None:
        func = lambda: obj
    else:
        func = partial(func, obj)

    return HashProxy(Factory(func), hash_func=hash_func, hash_value=hash_func(obj))


def register_backend():
    """Register proxy backend for use with joblib memory."""
    logger.debug("Registering proxy Joblib backend.")
    register_store_backend("proxy", ProxyStoreBackend)


class DeletedError(Exception):
    pass


class ProxyStoreBackend(CommonBackend):
    """A StoreBackend used with local or network file systems."""

    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    # Run outside of context managers.
    initial_hashers = [HashProxyHasher()] + list(_default_initial_hashers)
    # Context managers that temporarily change objects to enable consistent hashing.
    context_managers = list(_default_context_managers)
    # Run within context managers.
    guarded_hashers = list(_default_guarded_hashers)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []

        for dirpath, _, filenames in os.walk(self.location):
            is_cache_hash_dir = re.match("[a-f0-9]{32}", os.path.basename(dirpath))

            if is_cache_hash_dir:
                try:
                    for filename in ("output.cpkl", "hash_values.txt"):
                        output_filename = os.path.join(dirpath, filename)
                        try:
                            last_access = os.path.getatime(output_filename)
                        except OSError:
                            try:
                                last_access = os.path.getatime(dirpath)
                            except OSError:
                                # The directory has already been deleted
                                raise DeletedError()
                except DeletedError:
                    # The directory has already been deleted
                    continue

                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                    dirsize = sum(os.path.getsize(fn) for fn in full_filenames)
                except OSError:
                    # Either output_filename or one of the files in
                    # dirpath does not exist any more. We assume this
                    # directory is being cleaned by another process already
                    continue

                items.append(CacheItemInfo(dirpath, dirsize, last_access))

        return items

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend."""
        # setup location directory
        self.location = location
        if not os.path.exists(self.location):
            mkdirp(self.location)

        self.verbose = verbose

    def load_item(self, path, verbose=1, msg=None):
        """Load an item from the store given its path as a list of strings."""
        full_path = os.path.join(self.location, *path)

        if verbose > 1:
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, full_path))

        filename = os.path.join(full_path, "output.cpkl")
        if not self._item_exists(filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % filename
            )

        hash_filename = os.path.join(full_path, "hash_values.txt")
        if not self._item_exists(hash_filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % hash_filename
            )

        with open(hash_filename, "r") as f:
            hash_values = ast.literal_eval(f.read())

        logger.debug(f"Creating Factory from filename '{filename}'.")
        logger.debug(f"Loaded hash values: {hash_values}.")

        def load_func():
            if hasattr(load_func, "stored"):
                logger.debug("Returning stored data.")
                return load_func.stored

            logger.debug(f"cloudpickle loading filename '{filename}'.")
            with open(filename, "rb") as f:
                contents = cloudpickle.load(f)

            load_func.stored = contents
            return contents

        if len(hash_values) == 1:
            # If only a single value has been stored.
            return HashProxy(
                Factory(load_func), hash_func=self.get_hash, hash_value=hash_values[0]
            )
        # Otherwise create a lazy proxy for each individual object to associate each
        # stored object with its individual hash value.

        def get_factory_func(i):
            def factory_func():
                return load_func()[i]

            return factory_func

        return tuple(
            HashProxy(
                Factory(get_factory_func(i)),
                hash_func=self.get_hash,
                hash_value=hash_value,
            )
            for i, hash_value in enumerate(hash_values)
        )

    def dump_item(self, path, item, verbose=1):
        """Dump an item in the store at the path given as a list of
        strings."""
        try:
            item_path = os.path.join(self.location, *path)
            if not self._item_exists(item_path):
                self.create_location(item_path)
            filename = os.path.join(item_path, "output.cpkl")
            logger.debug(f"Caching '{item}' in '{item_path}'.")

            hash_filename = os.path.join(item_path, "hash_values.txt")
            logger.debug(f"Writing hash values to '{item_path}'.")

            if verbose > 10:
                print("Persisting in %s" % item_path)

            # If requested to do so, do not save the returned `item`. Only a
            # placeholder is saved instead in this case.
            if isinstance(item, tuple) and len(item) == 2 and item[1] is HASHES_ONLY:

                def hash_only_write_func(to_write, dest_filename):
                    # `to_write` is ignored.
                    with open(dest_filename, "w") as f:
                        f.write("")

                self._concurrency_safe_write(None, filename, hash_only_write_func)

                # Remove the HASHES_ONLY argument in order to allow the hashing below
                # to operate normally.
                item = item[0]
            else:
                # Save normally.

                def write_func(to_write, dest_filename):
                    with open(dest_filename, "wb") as f:
                        cloudpickle.dump(to_write, f, protocol=-1)

                self._concurrency_safe_write(item, filename, write_func)

            # Always write hash values as normal.

            def hash_write_func(to_write, hash_filename):
                if not isinstance(to_write, tuple):
                    to_write = (to_write,)

                with open(hash_filename, "w") as f:
                    f.write(str(list(map(self.get_hash, to_write))))

            self._concurrency_safe_write(item, hash_filename, hash_write_func)

        except:  # noqa: E722
            " Race condition in the creation of the directory "

    def contains_item(self, path):
        """Check if there is an item at the path, given as a list of strings"""
        item_path = os.path.join(self.location, *path)
        filename = os.path.join(item_path, "output.cpkl")
        hash_filename = os.path.join(item_path, "hash_values.txt")

        return self._item_exists(filename) and self._item_exists(hash_filename)
